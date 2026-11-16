import torch

from neverwake.world.memory import MemoryModel
from neverwake.world.next_obs_model import NextObsModel
from neverwake.world.reward_end_model import RewardEndModel


class BatchedWorldStepper:
    def __init__(
        self,
        ctx_frames: int,
        memory: MemoryModel,
        next_obs_model: NextObsModel,
        reward_end: RewardEndModel,
        obs_ctx_stack_init: torch.Tensor,
        action_seq_init: torch.Tensor,
    ):
        """
        obs_ctx_stack_init: (B, T, C, H, W)
        action_seq_init: (B, T-1)
        Assume all states are current valid

        Track the last T ctx for the denoiser. The actions only go through the
        memory: it starts by reading the first T-1 ctx with their actions and then
        every step it reads the last ctx with the new action

        The reward model only uses what the memory gives
        """
        if ctx_frames < 1:
            raise ValueError(f"ctx frames {ctx_frames} should be >= 1")
        if ctx_frames != obs_ctx_stack_init.shape[1]:
            raise ValueError(
                f"ctx frames {ctx_frames} != actual {obs_ctx_stack_init.shape[1]}"
            )

        self.obs_ctx_stack = obs_ctx_stack_init.clone()

        self.memory = memory
        self.next_obs_model = next_obs_model
        self.reward_end = reward_end

        self.device_type = obs_ctx_stack_init.device.type
        self.use_bf16 = self.device_type == "cuda"

        self.memory_state: tuple[torch.Tensor, torch.Tensor] | None = None

        with (
            torch.no_grad(),
            torch.autocast(self.device_type, torch.bfloat16, enabled=self.use_bf16),
        ):
            for idx in range(ctx_frames - 1):
                _, self.memory_state = self.memory.step(
                    self.obs_ctx_stack[:, idx, ...],
                    action_seq_init[:, idx],
                    self.memory_state,
                )

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        actions: (B,)
        """
        with (
            torch.no_grad(),
            torch.autocast(self.device_type, torch.bfloat16, enabled=self.use_bf16),
        ):
            # The memory reads the last ctx with the action taken on it
            memory_emb, self.memory_state = self.memory.step(
                self.obs_ctx_stack[:, -1, ...], actions, self.memory_state
            )

            # Then get the next state
            # (B, C, H, W)
            next_obs = self.next_obs_model.next_obs_sample(
                self.obs_ctx_stack, memory_emb
            )

            reward_logits, end_logits = self.reward_end(memory_emb)

        # Update
        self.obs_ctx_stack[:, :-1, ...] = self.obs_ctx_stack[:, 1:, ...].clone()
        self.obs_ctx_stack[:, -1, ...] = next_obs

        return next_obs.float(), reward_logits.float(), end_logits.float()
