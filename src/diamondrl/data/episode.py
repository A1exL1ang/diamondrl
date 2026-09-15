from dataclasses import dataclass

import torch


@dataclass
class RolloutBatch:
    """
    TODO: fix this docstring

    You can think of the first "ctx" things as "context/input" with the
    most recent state existing and padding for start is permitted. Then
    the state is rolled out for "roll" steps.

    More specifically, obs[:, :ctx] represent the "context/input"
    and obs[:, ctx - 1] must be real.

    Transition information is stored for all obs but the last
    but assume that the ones for padded obs are not relevant.

    This is a slight deviation from DIAMOND where it allows for
    the "roll" steps to be padding which might be good for short trajectories.
    """

    obs_stack: torch.Tensor  # (B, ctx + roll, C, H, W)
    actions: torch.Tensor  # (B, ctx + roll - 1)
    rewards: torch.Tensor  # (B, ctx + roll - 1)
    terminations: torch.Tensor  # (B, ctx + roll - 1)

    ctx: int

    @classmethod
    def cat(cls, batches: list["RolloutBatch"]) -> "RolloutBatch":
        assert batches, "batches should exist"
        assert all(b.ctx == batches[0].ctx for b in batches), "ctx should be consistent"

        return cls(
            obs_stack=torch.cat([b.obs_stack for b in batches]),
            actions=torch.cat([b.actions for b in batches]),
            rewards=torch.cat([b.rewards for b in batches]),
            terminations=torch.cat([b.terminations for b in batches]),
            ctx=batches[0].ctx,
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
    ):
        if not (
            len(actions) == len(rewards) == len(terminations) == len(observations) - 1
        ):
            raise ValueError("episode fields have inconsistent lengths")

        self.observations = torch.stack(observations).detach().cpu().clone()
        self.actions = torch.tensor(actions).long()
        self.rewards = torch.tensor(rewards).float()
        self.terminations = torch.tensor(terminations).bool()

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

        # (n, W, C, H, W) * (n, W, 1, 1, 1)
        obs_stack = (
            self.observations[safe_indices_obs] * good_obs[:, :, None, None, None]
        )

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

    def sample_rollout(
        self, batch_size: int, ctx_frames: int, roll_actions: int
    ) -> RolloutBatch:
        """
        think of roll_len as the number of actions u roll folwards
        ctx is the number of FRAMES and roll is number of actions past that one

        """
        if ctx_frames < 1:
            raise ValueError(f"ctx_len {ctx_frames} must be >= 1")
        if roll_actions < 0:
            raise ValueError(f"roll_len {roll_actions} must be >= 0")
        if len(self.actions) < max(roll_actions, 1):
            raise ValueError(
                f"have {len(self.actions)} transitions but need max({roll_actions}, 1)"
            )

        # Note that randint is exclusive and in the case roll_actions=0 we don't want to
        # sample the terminal state
        anchors = torch.randint(
            0,
            min(len(self.actions) - roll_actions + 1, len(self.actions)),
            (batch_size,),
        )
        offsets = torch.arange(-ctx_frames + 1, roll_actions + 1)

        # (n, 1) + (1, W) --> (n, W) where i-th row are raw indices for i-th window
        # Note that we will chop off final column for actions/rewards/terminations
        indices = anchors[:, None] + offsets[None, :]

        data = self.gather(indices)

        return RolloutBatch(
            obs_stack=data["obs_stack"],
            actions=data["actions"],
            rewards=data["rewards"],
            terminations=data["terminations"],
            ctx=ctx_frames,
        )

    def sample_sequence(
        self, batch_size: int, seq_transition_count: int
    ) -> MaskedSequenceBatch:
        if seq_transition_count < 1:
            raise ValueError(f"seq_len {seq_transition_count} must be >= 1")
        if len(self.actions) < 1:
            raise ValueError("must have at least one trans")

        # Sample a random transition and then the index of the transition
        starts = torch.randint(0, len(self.actions), (batch_size,)) - torch.randint(
            0, seq_transition_count, (batch_size,)
        )
        offsets = torch.arange(0, seq_transition_count + 1)
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
        self.observations: list[torch.Tensor] = [
            initial_observation.detach().cpu().clone()
        ]
        self.actions: list[int] = []
        self.rewards: list[float] = []
        self.terminations: list[bool] = []

    def push_info(
        self, action: int, reward: float, observation: torch.Tensor, termination: bool
    ) -> None:
        self.observations.append(observation.detach().cpu().clone())
        self.actions.append(action)
        self.rewards.append(reward)
        self.terminations.append(termination)

    def snapshot(self) -> Episode:
        return Episode(self.observations, self.actions, self.rewards, self.terminations)
