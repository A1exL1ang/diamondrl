import torch

from diamondrl.world.next_obs_model import NextObsModel
from diamondrl.world.reward_end_model import RewardEndModel


class BatchedWorldStepper:
    def __init__(
        self,
        ctx_frames: int,
        next_obs_model: NextObsModel,
        reward_end: RewardEndModel,
        obs_ctx_stack_init: torch.Tensor,
        action_seq_init: torch.Tensor,
    ):
        """
        obs_ctx_stack_init: (B, T, C, H, W)
        action_seq_init: (B, T-1)
        Assume all states are current valid

        Track last T ctx and last T-1 actions. When we query the obs model we
        pass in the T ctx and append the final action to form T actions

        The reward model only uses the last ctx, new ctx and new action
        """
        if ctx_frames < 1:
            raise ValueError(f"ctx frames {ctx_frames} should be >= 1")
        if ctx_frames != obs_ctx_stack_init.shape[1]:
            raise ValueError(
                f"ctx frames {ctx_frames} != actual {obs_ctx_stack_init.shape[1]}"
            )

        self.obs_ctx_stack = obs_ctx_stack_init.clone()
        self.action_seq = action_seq_init.clone()

        self.next_obs_model = next_obs_model
        self.reward_end = reward_end

        with torch.no_grad():
            self.lstm_state: tuple[torch.Tensor, torch.Tensor] | None = (
                self.reward_end(
                    self.obs_ctx_stack[:, :-1, ...],
                    self.action_seq,
                    self.obs_ctx_stack[:, 1:, ...],
                    None,
                )[2]
                if ctx_frames > 1
                else None
            )

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        actions: (B,)
        """
        # First get the next state
        # (B, C, H, W)
        with torch.no_grad():
            next_obs = self.next_obs_model.next_obs_sample(
                self.obs_ctx_stack,
                torch.cat([self.action_seq, actions.unsqueeze(1)], dim=1),
            )

        # Sequence size of 1
        with torch.no_grad():
            reward_logits, end_logits, self.lstm_state = self.reward_end(
                self.obs_ctx_stack[:, -1:, ...],
                actions.unsqueeze(1),
                next_obs.unsqueeze(1),
                self.lstm_state,
            )

        # Update
        self.obs_ctx_stack[:, :-1, ...] = self.obs_ctx_stack[:, 1:, ...].clone()
        self.obs_ctx_stack[:, -1, ...] = next_obs

        if self.action_seq.shape[1] > 0:
            self.action_seq[:, :-1] = self.action_seq[:, 1:].clone()
            self.action_seq[:, -1] = actions

        return next_obs, reward_logits.squeeze(1), end_logits.squeeze(1)
