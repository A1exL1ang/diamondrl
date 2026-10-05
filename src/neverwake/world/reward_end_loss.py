import torch
import torch.nn.functional as F

from neverwake.data.episode import MaskedSequenceBatch
from neverwake.world.reward_end_model import RewardEndModel


def reward_end_loss(
    reward_end: RewardEndModel, seq_batch: MaskedSequenceBatch
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    valid_mask_flat = seq_batch.valid.float().flatten()
    num_valid = valid_mask_flat.sum().clamp_min(1.0)

    reward_target_flat = seq_batch.rewards.flatten().sign().long() + 1
    end_target_flat = seq_batch.terminations.flatten().long()

    # Forward
    reward_logits, end_logits, _ = reward_end(
        seq_batch.obs_stack[:, :-1, ...],
        seq_batch.actions,
        seq_batch.obs_stack[:, 1:, ...],
        None,
    )

    # Loss of reward sign
    loss_reward = (
        F.cross_entropy(
            reward_logits.flatten(0, 1),
            reward_target_flat,
            reduction="none",
        )
        * valid_mask_flat
    ).sum() / num_valid

    # Loss of end
    loss_end = (
        F.cross_entropy(
            end_logits.flatten(0, 1),
            end_target_flat,
            reduction="none",
        )
        * valid_mask_flat
    ).sum() / num_valid

    # Total
    loss = loss_reward + loss_end

    return loss, {
        "loss_reward": loss_reward.detach(),
        "loss_end": loss_end.detach(),
    }
