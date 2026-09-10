import torch

from diamondrl.edm.denoiser import Denoiser


def sampler(
    denoiser: Denoiser,
    labels: torch.Tensor,
    img_shape: tuple[int, int, int] = (1, 28, 28),
    rho: float = 7.0,
    sigma_max: float = 80.0,
    sigma_min: float = 0.002,
    steps: int = 32,
) -> torch.Tensor:
    device = next(denoiser.parameters()).device
    x = torch.normal(
        mean=0.0, std=sigma_max, size=(labels.shape[0], *img_shape), device=device
    )

    sigmas = [
        (
            sigma_max ** (1 / rho)
            + i / (steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
        )
        ** rho
        for i in range(steps)
    ]
    sigmas = torch.tensor(sigmas + [0.0], device=device)

    for i in range(steps):
        t = sigmas[i]
        delta_t = sigmas[i + 1] - sigmas[i]

        d = 1 / t * (x - denoiser(x, sigmas[i].expand(labels.shape[0]), labels))
        x = x + delta_t * d

    return x
