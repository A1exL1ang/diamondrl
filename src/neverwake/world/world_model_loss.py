import torch

from neverwake.data.episode import MaskedSequenceBatch
from neverwake.world.denoiser_loss import denoiser_loss
from neverwake.world.reward_end_loss import reward_end_loss
from neverwake.world.world_model import WorldModel


def world_model_loss(
    world_model: WorldModel,
    seq_batch: MaskedSequenceBatch,
    sign_rewards: bool,
    ctx_frames: int,
    future_frames: int,
    sigma_min: float,
    sigma_max: float,
    p_mean: float,
    p_std: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    One pass of the memory over the sequence, then the reward / end loss on every
    transition and the denoiser loss on future_frames frames. Gradients from both
    reach the memory.

    seq_batch.obs_stack: (B, L + 1, C, H, W)
    seq_batch.actions: (B, L)
    """
    # (B, L, memory_dim) where entry t has seen frames and actions 0..t
    # Note: the final frame has no action so it is only ever a denoiser target
    memory_emb = world_model.memory(seq_batch.obs_stack[:, :-1, ...], seq_batch.actions)

    loss_reward, loss_end = reward_end_loss(
        world_model.reward_end, seq_batch, memory_emb, sign_rewards
    )
    loss_denoiser, info_denoiser = denoiser_loss(
        denoiser=world_model.denoiser,
        seq_batch=seq_batch,
        memory_emb=memory_emb,
        ctx_frames=ctx_frames,
        future_frames=future_frames,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        p_mean=p_mean,
        p_std=p_std,
    )

    # HarmonyDream (Ma et al. 2024): loss / scale + log(1 + scale), learned scale
    # Note: same as EDELINE the end loss stays at a weight of 1
    scale_reward = torch.exp(world_model.log_scale_reward)
    scale_denoiser = torch.exp(world_model.log_scale_denoiser)

    loss = (
        loss_reward / scale_reward
        + loss_denoiser / scale_denoiser
        + loss_end
        + torch.log(1 + scale_reward)
        + torch.log(1 + scale_denoiser)
    )

    return loss, {
        "loss_reward": loss_reward.detach(),
        "loss_end": loss_end.detach(),
        "loss_denoiser": loss_denoiser.detach(),
        "scale_reward": scale_reward.detach(),
        "scale_denoiser": scale_denoiser.detach(),
        **info_denoiser,
    }
