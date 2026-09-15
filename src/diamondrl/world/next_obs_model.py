import torch

from diamondrl.edm.denoiser import Denoiser
from diamondrl.edm.sampling import sampler


class NextObsModel:
    def __init__(
        self,
        denoiser: Denoiser,
        obs_shape: tuple[int, int, int],
        rho: float,
        sigma_max: float,
        sigma_min: float,
        steps: int,
    ):
        self.denoiser = denoiser
        self.obs_shape = obs_shape
        self.rho = rho
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.steps = steps

    def next_obs_sample(
        self, obs_ctx_stack: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """
        obs_ctx_buff: (B, T, C, H, W)
        """
        batch, seq, ch = obs_ctx_stack.shape[:3]
        obs_ctx_cat = obs_ctx_stack.reshape((batch, seq * ch, *obs_ctx_stack.shape[3:]))

        return sampler(
            self.denoiser,
            obs_ctx_cat,
            actions,
            self.obs_shape,
            self.rho,
            self.sigma_max,
            self.sigma_min,
            self.steps,
        )
