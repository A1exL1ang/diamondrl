import torch

from neverwake.data.episode import MaskedSequenceBatch
from neverwake.world.denoiser import Denoiser


def denoiser_loss_single(
    denoiser: Denoiser,
    obs_window: torch.Tensor,
    memory_emb: torch.Tensor,
    valid: torch.Tensor,
    sigma_min: float,
    sigma_max: float,
    p_mean: float,
    p_std: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    EDM loss with lambda = 1 / c_out^2 (unit weight on F) and log-normal sigma.

    obs_window: (B, T+1, C, H, W), T context frames then the target frame
    memory_emb: (B, memory_dim), has seen everything up to the last context frame
                and the action taken on it
    valid: (B,), False for the rows whose target frame is padding
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
    denoiser_out = denoiser(obs_ctx_cat, noisy_x, sigma, memory_emb)

    # Mean over each frame and then over the rows that have a real target
    loss_per_row = (
        lambda_weight[:, None, None, None] * (denoiser_out - obs_targ) ** 2
    ).mean(dim=(1, 2, 3))
    loss = (loss_per_row * valid).sum() / valid.sum().clamp_min(1)

    return loss, denoiser_out


def denoiser_loss(
    denoiser: Denoiser,
    seq_batch: MaskedSequenceBatch,
    memory_emb: torch.Tensor,
    ctx_frames: int,
    future_frames: int,
    sigma_min: float,
    sigma_max: float,
    p_mean: float,
    p_std: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Predicts future_frames frames in a row at one random spot of the sequence (the
    same spot for every row); predictions are written back (detached and clamped)
    as context for the next future frame.

    seq_batch.obs_stack: (B, L + 1, C, H, W)
    memory_emb: (B, L, memory_dim) where entry t has seen frames and actions 0..t
    """
    obs_stack = seq_batch.obs_stack.clone()
    num_frames = obs_stack.shape[1]

    if num_frames < ctx_frames + future_frames:
        raise ValueError(
            f"have {num_frames} frames but need {ctx_frames} + {future_frames}"
        )

    # First context frame of the first prediction
    start = int(torch.randint(0, num_frames - (ctx_frames + future_frames) + 1, ()))

    losses: list[torch.Tensor] = []

    for frame in range(start, start + future_frames):
        targ = frame + ctx_frames

        # The target is real if the transition into it is, and that transition is
        # also the last thing memory_emb[:, targ - 1] has seen
        valid = seq_batch.valid[:, targ - 1]

        # Note: cloned so the compiled denoiser gets a plain input. A view into the
        # compiled memory's output sits at a different address for every start and
        # cudagraphs would record each of those separately
        loss, denoiser_out = denoiser_loss_single(
            denoiser=denoiser,
            obs_window=obs_stack[:, frame : targ + 1, ...],
            memory_emb=memory_emb[:, targ - 1, ...].clone(),
            valid=valid,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            p_mean=p_mean,
            p_std=p_std,
        )

        # Note: only written back where the target is real. A row that is padding
        # here can hold the real 1st frame of its episode, which the next future
        # frame needs as context
        obs_stack[:, targ, ...] = torch.where(
            valid[:, None, None, None],
            denoiser_out.detach().clamp(min=-1.0, max=+1.0),
            obs_stack[:, targ, ...],
        )
        losses.append(loss)

    info = {f"loss_frame_{i}": loss.detach() for i, loss in enumerate(losses)}
    return torch.stack(losses).mean(), info
