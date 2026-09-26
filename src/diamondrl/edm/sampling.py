import torch
from torch._tensor import Tensor

from diamondrl.edm.denoiser import Denoiser


def sampler(
    denoiser: Denoiser,
    obs_ctx_cat: torch.Tensor,
    actions: torch.Tensor,
    obs_shape: tuple[int, int, int],
    rho: float,
    sigma_max_ode: float,
    sigma_min_ode: float,
    steps: int,
) -> torch.Tensor:
    # TODO: swap sigma max min

    """
    obs_shape is for a single obs
    """
    if steps <= 0:
        raise ValueError(f"steps {steps} should be > 0")

    device = next(denoiser.parameters()).device
    x: Tensor = torch.normal(
        mean=0.0,
        std=sigma_max_ode,
        size=(obs_ctx_cat.shape[0], *obs_shape),
        device=device,
    )

    sigmas = (
        [
            (
                sigma_max_ode ** (1 / rho)
                + i
                / (steps - 1)
                * (sigma_min_ode ** (1 / rho) - sigma_max_ode ** (1 / rho))
            )
            ** rho
            for i in range(steps)
        ]
        if steps > 1
        else [sigma_max_ode]
    )
    sigmas = torch.tensor(sigmas + [0.0], device=device)

    for i in range(steps):
        t = sigmas[i]
        delta_t = sigmas[i + 1] - sigmas[i]

        current_cond_sigma = sigmas[i].expand(obs_ctx_cat.shape[0])
        d = 1 / t * (x - denoiser(obs_ctx_cat, x, current_cond_sigma, actions))

        x = x + delta_t * d

    return x
