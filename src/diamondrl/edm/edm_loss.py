import torch

from diamondrl.data.episode import ContextBatch
from diamondrl.edm.denoiser import Denoiser


def edm_loss_single(
    denoiser: Denoiser,
    obs_ctx_stk_with_next: torch.Tensor,
    actions: torch.Tensor,
    sigma_min_train: float,
    sigma_max_train: float,
    p_mean: float,
    p_std: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    obs_ctx_stk_with_next: (B, T+1, C, H, W)
    actions: (B, T)
    """
    device = next(denoiser.parameters()).device
    batch_size, _, frame_ch, frame_h, frame_w = obs_ctx_stk_with_next.shape

    # (B, T, C, H, W) --> (B, nC, H, W) for both (with different C)
    obs_ctx_cat = obs_ctx_stk_with_next[:, :-1, ...].flatten(1, 2)
    obs_targ = obs_ctx_stk_with_next[:, -1:, ...].flatten(1, 2)

    # The original EDM sigma
    sigma_original = torch.exp(
        torch.normal(mean=p_mean, std=p_std, size=(batch_size,), device=device)
    ).clamp(min=sigma_min_train, max=sigma_max_train)
    sigma_true = torch.sqrt(sigma_original**2 + denoiser.sigma_off**2)

    # Make effective weight 1
    lambda_weight = (sigma_true**2 + denoiser.sigma_data**2) / (
        (sigma_true * denoiser.sigma_data) ** 2
    )

    # Create the random noise and add it to get sample at spot on prob path (B, C, H, W)
    noisy_x = sigma_original[:, None, None, None] * torch.randn(
        (batch_size, frame_ch, frame_h, frame_w), device=device
    )
    noisy_x += denoiser.sigma_off * torch.randn(
        (batch_size, frame_ch, 1, 1), device=device
    )
    noisy_x += obs_targ

    # First forward
    denoiser_out = denoiser(obs_ctx_cat, noisy_x, sigma_original, actions)

    loss = (lambda_weight[:, None, None, None] * (denoiser_out - obs_targ) ** 2).mean()

    return loss, denoiser_out


def edm_loss(
    denoiser: Denoiser,
    context_batch: ContextBatch,
    sigma_min_train: float,
    sigma_max_train: float,
    p_mean: float,
    p_std: float,
) -> tuple[torch.Tensor, list[float]]:
    """
    cur_frames_stk: (B, ctx + future_frames, C, H, W)
    actions: (B, ctx + extra_unroll_steps)
    """
    cur_frames_stk = context_batch.obs_stack.clone()
    ctx = context_batch.ctx
    future_frames = context_batch.obs_stack.shape[1] - ctx

    losses: list[torch.Tensor] = []

    # First step doesn't unroll
    for step in range(future_frames):
        loss, denoiser_out = edm_loss_single(
            denoiser=denoiser,
            obs_ctx_stk_with_next=cur_frames_stk[:, step : step + ctx + 1, ...],
            actions=context_batch.actions[:, step : step + ctx],
            sigma_min_train=sigma_min_train,
            sigma_max_train=sigma_max_train,
            p_mean=p_mean,
            p_std=p_std,
        )
        cur_frames_stk[:, step + ctx, ...] = denoiser_out.detach().clamp(
            min=-1.0, max=+1.0
        )
        losses.append(loss)

    stacked = torch.stack(losses)
    return stacked.mean(), stacked.tolist()
