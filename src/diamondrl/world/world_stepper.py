import torch

from diamondrl.models.reward_end_model import RewardEndModel
from diamondrl.world.next_obs_model import NextObsModel


class BatchedWorldStepper:
    def __init__(
        self,
        ctx_frames: int,
        next_obs_model: NextObsModel,
        reward_end_model: RewardEndModel,
        obs_ctx_stk_init: torch.Tensor,
        action_seq_init: torch.Tensor,
    ):
        """
        obs_ctx_stk: (B, T, C, H, W)
        action_seq_init: (B, T-1)
        Assume all states are current valid

        Track last T ctx and last T-1 actions. When we query the obs model we
        pass in the T ctx and append the final action to form T actions

        The reward model only uses the last ctx, new ctx and new action
        """
        if ctx_frames < 1:
            raise ValueError(f"ctx frames {ctx_frames} should be >= 1")
        if ctx_frames != obs_ctx_stk_init.shape[1]:
            raise ValueError(
                f"ctx frames {ctx_frames} != actual {obs_ctx_stk_init.shape[1]}"
            )

        self.obs_ctx_stk = obs_ctx_stk_init.clone()
        self.actions_seq = action_seq_init.clone()

        self.next_obs_model = next_obs_model
        self.reward_end_model = reward_end_model

        with torch.no_grad():
            self.lstm_hidden: tuple[torch.Tensor, torch.Tensor] | None = (
                self.reward_end_model(
                    self.obs_ctx_stk[:, :-1, ...],
                    self.actions_seq,
                    self.obs_ctx_stk[:, 1:, ...],
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
                self.obs_ctx_stk,
                torch.cat([self.actions_seq, actions.unsqueeze(1)], dim=1),
            )

        # Sequence size of 1
        with torch.no_grad():
            reward_logits, end_logits, self.lstm_hidden = self.reward_end_model(
                self.obs_ctx_stk[:, -1:, ...],
                actions.unsqueeze(1),
                next_obs.unsqueeze(1),
                self.lstm_hidden,
            )

        # Update
        self.obs_ctx_stk[:, :-1, ...] = self.obs_ctx_stk[:, 1:, ...].clone()
        self.obs_ctx_stk[:, -1, ...] = next_obs

        if self.actions_seq.shape[1] > 0:
            self.actions_seq[:, :-1] = self.actions_seq[:, 1:].clone()
            self.actions_seq[:, -1] = actions

        return next_obs, reward_logits.squeeze(1), end_logits.squeeze(1)
