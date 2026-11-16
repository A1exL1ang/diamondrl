import torch
import torch.nn as nn


class FourierEncoding(nn.Module):
    freqs: torch.Tensor

    def __init__(self, emb_dim: int, w_min: float, w_max: float):
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

        return emb


class CondEmbedder(nn.Module):
    def __init__(self, cond_dim: int, memory_dim: int, w_min: float, w_max: float):
        super().__init__()
        if cond_dim % 2 != 0:
            raise ValueError(f"cond_dim {cond_dim} should be even")

        self.sigma_emb = FourierEncoding(cond_dim, w_min, w_max)
        self.memory_proj = nn.Linear(memory_dim, cond_dim)

        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim)
        )

    def forward(
        self, sigma_transformed: torch.Tensor, memory_emb: torch.Tensor
    ) -> torch.Tensor:
        # Sigma transformed is (B,) and memory_emb is (B, memory_dim)
        # Note: no action embedding since the memory has already seen the actions
        cond_emb = self.sigma_emb(sigma_transformed) + self.memory_proj(memory_emb)
        cond_emb = self.cond_mlp(cond_emb)
        return cond_emb
