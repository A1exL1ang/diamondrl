import torch

from neverwake.data.episode import ContextBatch
from neverwake.world.denoiser import Denoiser


def denoiser_loss_single(
    denoiser: Denoiser,
    obs_window: torch.Tensor,
    actions: torch.Tensor,
    sigma_min: float,
    sigma_max: float,
    p_mean: float,
    p_std: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    EDM loss with lambda = 1 / c_out^2 (unit weight on F) and log-normal sigma.

    obs_window: (B, T+1, C, H, W), T context frames then the target frame
    actions: (B, T)
    """
    device = next(denoiser.parameters()).device
    batch_size, _, frame_ch, frame_h, frame_w = obs_window.shape

    # (B, T, C, H, W) --> (B, nC, H, W) for both (with different C)
    obs_ctx_cat = obs_window[:, :-1, ...].flatten(1, 2)
    obs_targ = obs_window[:, -1:, ...].flatten(1, 2)

    # The original EDM sigma
    sigma = torch.exp(
        torch.normal(mean=p_mean, std=p_std, size=(batch_size,), device=device)
    ).clamp(min=sigma_min, max=sigma_max)
    sigma_true = torch.sqrt(sigma**2 + denoiser.sigma_off**2)

    # Make effective weight 1
    lambda_weight = (sigma_true**2 + denoiser.sigma_data**2) / (
        (sigma_true * denoiser.sigma_data) ** 2
    )

    # Create the random noise and add it to get sample at spot on prob path (B, C, H, W)
    noisy_x = sigma[:, None, None, None] * torch.randn(
        (batch_size, frame_ch, frame_h, frame_w), device=device
    )
    noisy_x += denoiser.sigma_off * torch.randn(
        (batch_size, frame_ch, 1, 1), device=device
    )
    noisy_x += obs_targ

    # First forward
    denoiser_out = denoiser(obs_ctx_cat, noisy_x, sigma, actions)

    loss = (lambda_weight[:, None, None, None] * (denoiser_out - obs_targ) ** 2).mean()

    return loss, denoiser_out


def denoiser_loss(
    denoiser: Denoiser,
    context_batch: ContextBatch,
    sigma_min: float,
    sigma_max: float,
    p_mean: float,
    p_std: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Predicts each future frame autoregressively; predictions are written back
    (detached and clamped) as context for the next future frame.

    context_batch.obs_stack: (B, ctx + future_frames, C, H, W)
    context_batch.actions: (B, ctx + future_frames - 1)
    """
    obs_stack = context_batch.obs_stack.clone()
    ctx = context_batch.ctx
    future_frames = context_batch.obs_stack.shape[1] - ctx

    losses: list[torch.Tensor] = []

    for frame in range(future_frames):
        loss, denoiser_out = denoiser_loss_single(
            denoiser=denoiser,
            obs_window=obs_stack[:, frame : frame + ctx + 1, ...],
            actions=context_batch.actions[:, frame : frame + ctx],
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            p_mean=p_mean,
            p_std=p_std,
        )
        obs_stack[:, frame + ctx, ...] = denoiser_out.detach().clamp(min=-1.0, max=+1.0)
        losses.append(loss)

    info = {f"loss_frame_{i}": loss.detach() for i, loss in enumerate(losses)}
    return torch.stack(losses).mean(), info
