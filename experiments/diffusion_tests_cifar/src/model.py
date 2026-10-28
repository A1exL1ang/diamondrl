import torch
import torch.nn as nn
from emb import (
    CondEmbedderSigmaAndDiff,
    CondEmbedderSigmaAndJump,
    CondEmbedderSigmaOnly,
)
from unet import Unet


class Denoiser(nn.Module):
    sigma_data: torch.Tensor

    def __init__(
        self,
        net: Unet,
        cond_embedder: CondEmbedderSigmaOnly,
        sigma_data: float,
    ):
        super().__init__()
        self.cond_embedder = cond_embedder
        self.net = net  # F_theta in EDM

        self.register_buffer("sigma_data", torch.tensor(sigma_data))

    def forward(self, noisy_x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        # Important
        sigma_transformed = 0.25 * torch.log(sigma)

        c_in = 1 / torch.sqrt(self.sigma_data**2 + sigma**2)
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = (sigma * self.sigma_data) / torch.sqrt(self.sigma_data**2 + sigma**2)

        cond_emb = self.cond_embedder(sigma_transformed)

        # Make sure to avoid the bug of scaling what you condition on as well
        # We shouldn't scale obs_ctx by c_in but rather sigma_data for unit variance
        f_out = self.net(
            noisy_x * c_in[:, None, None, None],
            cond_emb,
        )
        return (
            c_skip[:, None, None, None] * noisy_x + c_out[:, None, None, None] * f_out
        )


class ShortcutModel(nn.Module):
    def __init__(self, net: Unet, cond_embedder: CondEmbedderSigmaAndJump):
        super().__init__()
        self.cond_embedder = cond_embedder
        self.net = net

    def forward(
        self, x_t: torch.Tensor, t: torch.Tensor, jump: torch.Tensor
    ) -> torch.Tensor:
        return self.net(x_t, self.cond_embedder(t, jump))


class MeanFlowModel(nn.Module):
    def __init__(self, net: Unet, cond_embedder: CondEmbedderSigmaAndDiff):
        super().__init__()
        self.cond_embedder = cond_embedder
        self.net = net

    def forward(
        self, x_t: torch.Tensor, t: torch.Tensor, diff: torch.Tensor
    ) -> torch.Tensor:
        return self.net(x_t, self.cond_embedder(t, diff))
