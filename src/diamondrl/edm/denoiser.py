import torch
import torch.nn as nn

from diamondrl.models.embedding import CondEmbedder
from diamondrl.models.unet import Unet


class Denoiser(nn.Module):
    sigma_off: torch.Tensor
    sigma_data: torch.Tensor

    def __init__(
        self,
        edm_f_net: Unet,
        cond_embedder: CondEmbedder,
        sigma_off: float,
        sigma_data: float,
    ):
        super().__init__()
        self.cond_embedder = cond_embedder
        self.edm_f_net = edm_f_net

        self.register_buffer("sigma_data", torch.tensor(sigma_data))
        self.register_buffer("sigma_off", torch.tensor(sigma_off))

    def forward(
        self,
        obs_ctx_cat: torch.Tensor,
        noisy_x: torch.Tensor,
        sigma_original: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        # Combine with sigma_off to get the real sigma and then rescale
        sigma_true = torch.sqrt(sigma_original**2 + self.sigma_off**2)
        sigma_transformed = 0.25 * torch.log(sigma_true)

        c_in = 1 / torch.sqrt(self.sigma_data**2 + sigma_true**2)
        c_skip = self.sigma_data**2 / (sigma_true**2 + self.sigma_data**2)
        c_out = (sigma_true * self.sigma_data) / torch.sqrt(
            self.sigma_data**2 + sigma_true**2
        )

        cond_emb = self.cond_embedder(sigma_transformed, actions)

        # Make sure to avoid the bug of scaling what you condition on as well
        # We shouldn't scale obs_ctx by c_in but rather sigma_data for unit variance
        f_out = self.edm_f_net(
            torch.cat(
                [obs_ctx_cat / self.sigma_data, noisy_x * c_in[:, None, None, None]],
                dim=1,
            ),
            cond_emb,
        )
        return (
            c_skip[:, None, None, None] * noisy_x + c_out[:, None, None, None] * f_out
        )
