import torch
import torch.nn as nn

from diamondrl.models.embedding import CondEmbedder
from diamondrl.models.unet import Unet


class Denoiser(nn.Module):
    sigma_data: torch.Tensor

    def __init__(self, edm_f_net: Unet, cond_embedder: CondEmbedder, sigma_data: float):
        super().__init__()
        self.cond_embedder = cond_embedder
        self.edm_f_net = edm_f_net
        self.register_buffer("sigma_data", torch.tensor(sigma_data))

    def forward(
        self, x: torch.Tensor, sigma_true: torch.Tensor, number: torch.Tensor
    ) -> torch.Tensor:
        sigma_transformed = 0.25 * torch.log(sigma_true)

        c_in = 1 / torch.sqrt(self.sigma_data**2 + sigma_true**2)
        c_skip = self.sigma_data**2 / (sigma_true**2 + self.sigma_data**2)
        c_out = (sigma_true * self.sigma_data) / torch.sqrt(
            self.sigma_data**2 + sigma_true**2
        )

        cond_emb = self.cond_embedder(sigma_transformed, number)
        f_out = self.edm_f_net(x * c_in[:, None, None, None], cond_emb)

        return c_skip[:, None, None, None] * x + c_out[:, None, None, None] * f_out
