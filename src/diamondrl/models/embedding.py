import torch
import torch.nn as nn


class FourierEncoding(nn.Module):
    freqs: torch.Tensor

    def __init__(self, emb_dim: int, w_min: float = 0.2, w_max: float = 15.0):
        super().__init__()
        freqs = (
            (2 * torch.pi)
            * w_min
            * torch.pow(w_max / w_min, torch.linspace(0, 1, emb_dim // 2))
        )
        self.register_buffer("freqs", freqs)
        self.emb_dim = emb_dim

    def forward(self, sigma_transformed: torch.Tensor) -> torch.Tensor:
        # After reshaping: (B, 1) x (1, emb // 2) --> (B, emb)
        emb = torch.cat(
            [
                torch.cos(sigma_transformed[:, None] * self.freqs[None, :]),
                torch.sin(sigma_transformed[:, None] * self.freqs[None, :]),
            ],
            dim=-1,
        )

        return emb * ((2 / self.emb_dim) ** 0.5)


class CondEmbedder(nn.Module):
    def __init__(self, cond_dim: int, ctx_steps: int, num_actions: int):
        super().__init__()
        if cond_dim % 2 != 0:
            raise ValueError(f"cond_dim {cond_dim} should be even")
        if cond_dim % ctx_steps != 0:
            raise ValueError(
                f"cond_dim {cond_dim} should be a multiple of ctx_steps {ctx_steps}"
            )

        self.sigma_emb = FourierEncoding(cond_dim)
        self.action_emb = nn.Embedding(num_actions, cond_dim // ctx_steps)

        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim)
        )

    def forward(
        self, sigma_transformed: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        # Sigma transformed and number are (B,)
        cond_emb = self.sigma_emb(sigma_transformed) + self.action_emb(actions).flatten(
            start_dim=1
        )
        cond_emb = self.cond_mlp(cond_emb)
        return cond_emb
