import torch
import torch.nn.functional as F

from neverwake.data.episode import MaskedSequenceBatch
from neverwake.world.reward_end_model import RewardEndModel


def reward_end_loss(
    reward_end: RewardEndModel,
    seq_batch: MaskedSequenceBatch,
    memory_emb: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    memory_emb: (B, L, memory_dim) where entry t has seen frames and actions 0..t, so
                it predicts the reward and end of transition t

    Returns the reward loss and the end loss, kept apart since the world model loss
    weights them differently
    """
    valid_mask_flat = seq_batch.valid.float().flatten()
    num_valid = valid_mask_flat.sum().clamp_min(1.0)

    reward_target_flat = seq_batch.rewards.flatten().sign().long() + 1
    end_target_flat = seq_batch.terminations.flatten().long()

    # Forward
    reward_logits, end_logits = reward_end(memory_emb)

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

    return loss_reward, loss_end
