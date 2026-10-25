from dataclasses import dataclass

import torch


@dataclass
class ContextBatch:
    """
    ctx context frames followed by future frames.

    The last context frame and every future frame are real; only the first
    ctx - 1 frames may be left padding (zero frames with action 0).
    actions[:, t] is the action taken at obs_stack[:, t].
    """

    obs_stack: torch.Tensor  # (B, ctx + future_frames, C, H, W)
    actions: torch.Tensor  # (B, ctx + future_frames - 1)
    rewards: torch.Tensor  # (B, ctx + future_frames - 1)
    terminations: torch.Tensor  # (B, ctx + future_frames - 1)

    ctx: int

    @classmethod
    def cat(cls, batches: list["ContextBatch"]) -> "ContextBatch":
        assert batches, "batches should exist"
        assert all(b.ctx == batches[0].ctx for b in batches), "ctx should be consistent"

        return cls(
            obs_stack=torch.cat([b.obs_stack for b in batches]),
            actions=torch.cat([b.actions for b in batches]),
            rewards=torch.cat([b.rewards for b in batches]),
            terminations=torch.cat([b.terminations for b in batches]),
            ctx=batches[0].ctx,
        )

    def to(self, device: torch.device | str) -> "ContextBatch":
        return ContextBatch(
            obs_stack=self.obs_stack.to(device),
            actions=self.actions.to(device),
            rewards=self.rewards.to(device),
            terminations=self.terminations.to(device),
            ctx=self.ctx,
        )


@dataclass
class MaskedSequenceBatch:
    """
    Supports left padding and suffix is not guarenteed to be valid
    """

    obs_stack: torch.Tensor  # (B, L + 1, C, H, W)
    actions: torch.Tensor  # (B, L)
    rewards: torch.Tensor  # (B, L)
    terminations: torch.Tensor  # (B, L)
    valid: torch.Tensor  # (B, L)

    @classmethod
    def cat(cls, batches: list["MaskedSequenceBatch"]) -> "MaskedSequenceBatch":
        assert batches, "batches should exist"

        return cls(
            obs_stack=torch.cat([b.obs_stack for b in batches]),
            actions=torch.cat([b.actions for b in batches]),
            rewards=torch.cat([b.rewards for b in batches]),
            terminations=torch.cat([b.terminations for b in batches]),
            valid=torch.cat([b.valid for b in batches]),
        )

    def to(self, device: torch.device | str) -> "MaskedSequenceBatch":
        return MaskedSequenceBatch(
            obs_stack=self.obs_stack.to(device),
            actions=self.actions.to(device),
            rewards=self.rewards.to(device),
            terminations=self.terminations.to(device),
            valid=self.valid.to(device),
        )


class Episode:
    """
    Finalized episode on cpu
    """

    def __init__(
        self,
        observations: list[torch.Tensor],
        actions: list[int],
        rewards: list[float],
        terminations: list[bool],
        epochs: list[int],
    ):
        if not (
            len(actions)
            == len(rewards)
            == len(terminations)
            == len(epochs)
            == len(observations) - 1
        ):
            raise ValueError("episode fields have inconsistent lengths")

        # Note: uint8 from EpisodeBuilder to save memory
        self.observations = torch.stack(observations).detach().cpu()
        self.actions = torch.tensor(actions).long()
        self.rewards = torch.tensor(rewards).float()
        self.terminations = torch.tensor(terminations).bool()
        self.epochs = torch.tensor(epochs).long()

    def gather(self, indices: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        indices represent the frames and includes the final one
        """
        good_obs = (indices >= 0) & (indices < len(self.observations))
        good_transition = (indices[:, :-1] >= 0) & (indices[:, :-1] < len(self.actions))

        safe_indices_obs = indices.clamp(min=0, max=len(self.observations) - 1)
        safe_indices_transition = indices[:, :-1].clamp(
            min=0, max=len(self.actions) - 1
        )

        # (n, W, C, H, W) * (n, W, 1, 1, 1) after going back to [-1, +1]
        obs_stack = (
            self.observations[safe_indices_obs].float() / (255 / 2) - 1
        ) * good_obs[:, :, None, None, None]

        # (n, W-1) * (n, W-1) with invalid ones zeroed out
        actions = self.actions[safe_indices_transition] * good_transition
        rewards = self.rewards[safe_indices_transition] * good_transition
        terminations = self.terminations[safe_indices_transition] * good_transition

        return {
            "obs_stack": obs_stack,
            "actions": actions,
            "rewards": rewards,
            "terminations": terminations,
            "valid": good_transition,
        }

    def sample_context(
        self, batch_size: int, ctx_frames: int, future_frames: int
    ) -> ContextBatch:
        """
        ctx_frames context frames followed by future_frames frames that must be real.
        With future_frames=0 the last context frame is never the episode's final frame.
        """
        if ctx_frames < 1:
            raise ValueError(f"ctx_frames {ctx_frames} must be >= 1")
        if future_frames < 0:
            raise ValueError(f"future_frames {future_frames} must be >= 0")
        if len(self.actions) < max(future_frames, 1):
            raise ValueError(
                f"have {len(self.actions)} transitions but need max({future_frames}, 1)"
            )

        # Note that randint is exclusive and in the case future_frames=0 we don't want
        # to sample the terminal state
        anchors = torch.randint(
            0,
            min(len(self.actions) - future_frames + 1, len(self.actions)),
            (batch_size,),
        )
        offsets = torch.arange(-ctx_frames + 1, future_frames + 1)

        # (n, 1) + (1, W) --> (n, W) where i-th row are raw indices for i-th window
        # Note that we will chop off final column for actions/rewards/terminations
        indices = anchors[:, None] + offsets[None, :]

        data = self.gather(indices)

        return ContextBatch(
            obs_stack=data["obs_stack"],
            actions=data["actions"],
            rewards=data["rewards"],
            terminations=data["terminations"],
            ctx=ctx_frames,
        )

    def sample_masked_sequence(
        self, batch_size: int, num_transitions: int
    ) -> MaskedSequenceBatch:
        if num_transitions < 1:
            raise ValueError(f"num_transitions {num_transitions} must be >= 1")
        if len(self.actions) < 1:
            raise ValueError("must have at least one trans")

        # Sample a random transition and then the index of the transition
        starts = torch.randint(0, len(self.actions), (batch_size,)) - torch.randint(
            0, num_transitions, (batch_size,)
        )
        offsets = torch.arange(0, num_transitions + 1)
        indices = starts[:, None] + offsets[None, :]

        data = self.gather(indices)

        return MaskedSequenceBatch(
            obs_stack=data["obs_stack"],
            actions=data["actions"],
            rewards=data["rewards"],
            terminations=data["terminations"],
            valid=data["valid"],
        )


class EpisodeBuilder:
    """
    Builds episode on CPU
    """

    def __init__(
        self,
        initial_observation: torch.Tensor,
    ) -> None:
        self.observations: list[torch.Tensor] = [self.quantize(initial_observation)]
        self.actions: list[int] = []
        self.rewards: list[float] = []
        self.terminations: list[bool] = []
        self.epochs: list[int] = []

    def push_info(
        self,
        action: int,
        reward: float,
        observation: torch.Tensor,
        termination: bool,
        epoch: int,
    ) -> None:
        self.observations.append(self.quantize(observation))
        self.actions.append(action)
        self.rewards.append(reward)
        self.terminations.append(termination)
        self.epochs.append(epoch)

    def quantize(self, observation: torch.Tensor) -> torch.Tensor:
        # [-1, +1] --> uint8 to save memory
        return (
            ((observation.detach().cpu() + 1) * (255 / 2)).round().clamp(0, 255).byte()
        )

    def snapshot(self) -> Episode:
        return Episode(
            self.observations,
            self.actions,
            self.rewards,
            self.terminations,
            self.epochs,
        )
