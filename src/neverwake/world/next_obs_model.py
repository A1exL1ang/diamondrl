import torch

from neverwake.world.denoiser import Denoiser
from neverwake.world.sampler import sampler


class NextObsModel:
    def __init__(
        self,
        denoiser: Denoiser,
        obs_shape: tuple[int, int, int],
        rho: float,
        sigma_min_ode: float,
        sigma_max_ode: float,
        denoising_steps: int,
    ):
        self.denoiser = denoiser
        self.obs_shape = obs_shape
        self.rho = rho
        self.sigma_min_ode = sigma_min_ode
        self.sigma_max_ode = sigma_max_ode
        self.denoising_steps = denoising_steps

    def next_obs_sample(
        self, obs_ctx_stack: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """
        obs_ctx_stack: (B, T, C, H, W)
        actions: (B, T)
        """
        obs_ctx_cat = obs_ctx_stack.flatten(1, 2)

        return sampler(
            self.denoiser,
            obs_ctx_cat,
            actions,
            self.obs_shape,
            self.rho,
            self.sigma_min_ode,
            self.sigma_max_ode,
            self.denoising_steps,
        ).clamp(min=-1.0, max=+1.0)
