import torch

from neverwake.world.denoiser import Denoiser


def sampler(
    denoiser: Denoiser,
    obs_ctx_cat: torch.Tensor,
    actions: torch.Tensor,
    obs_shape: tuple[int, int, int],
    rho: float,
    sigma_min_ode: float,
    sigma_max_ode: float,
    denoising_steps: int,
) -> torch.Tensor:
    """
    EDM rho-schedule (Karras et al. 2022, eq. 5) with Euler steps.

    obs_shape is for a single obs
    """
    if denoising_steps <= 0:
        raise ValueError(f"denoising_steps {denoising_steps} should be > 0")

    device = next(denoiser.parameters()).device
    x: torch.Tensor = torch.normal(
        mean=0.0,
        std=sigma_max_ode,
        size=(obs_ctx_cat.shape[0], *obs_shape),
        device=device,
    )

    # We replace the last point of sigma_min with 0 instead since 0.002 -> 0 is wasteful
    sigmas = [
        (
            sigma_max_ode ** (1 / rho)
            + i
            / denoising_steps
            * (sigma_min_ode ** (1 / rho) - sigma_max_ode ** (1 / rho))
        )
        ** rho
        for i in range(denoising_steps)
    ]
    sigmas = torch.tensor(sigmas + [0.0], device=device)

    for i in range(denoising_steps):
        t = sigmas[i]
        delta_t = sigmas[i + 1] - sigmas[i]

        current_cond_sigma = sigmas[i].expand(obs_ctx_cat.shape[0])
        d = 1 / t * (x - denoiser(obs_ctx_cat, x, current_cond_sigma, actions))

        x = x + delta_t * d

    return x
