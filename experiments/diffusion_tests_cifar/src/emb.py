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

        return emb * ((2 / self.emb_dim) ** 0.5)


class CondEmbedderSigmaOnly(nn.Module):
    def __init__(self, cond_dim: int, w_min: float, w_max: float):
        super().__init__()

        self.sigma_emb = FourierEncoding(cond_dim, w_min, w_max)

        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim)
        )

    def forward(self, sigma_transformed: torch.Tensor) -> torch.Tensor:
        # Sigma transformed and number are (B,)
        cond_emb = self.sigma_emb(sigma_transformed)
        cond_emb = self.cond_mlp(cond_emb)
        return cond_emb


class CondEmbedderSigmaAndJump(nn.Module):
    def __init__(
        self,
        cond_dim: int,
        w_min_sigma: float,
        w_max_sigma: float,
        num_jumps: int,
    ):
        super().__init__()

        self.num_jumps = num_jumps

        self.sigma_emb = FourierEncoding(cond_dim, w_min_sigma, w_max_sigma)

        # Jumps 1, 1/2, ..., 2^-(num_jumps - 1) plus a jump of 0
        self.jump_emb = nn.Embedding(num_jumps + 1, cond_dim)

        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim)
        )

    def forward(
        self, sigma_transformed: torch.Tensor, jump: torch.Tensor
    ) -> torch.Tensor:
        # Sigma transformed and jump are (B,), jump is a fraction of the schedule

        # Note: 1 maps to num_jumps and anything below the finest jump maps to 0
        jump_idx = self.num_jumps + torch.log2(jump.clamp(min=2.0**-self.num_jumps))

        cond_emb = self.sigma_emb(sigma_transformed) + self.jump_emb(
            jump_idx.round().long()
        )
        cond_emb = self.cond_mlp(cond_emb)
        return cond_emb


class CondEmbedderSigmaAndDiff(nn.Module):
    def __init__(
        self,
        cond_dim: int,
        w_min_sigma: float,
        w_max_sigma: float,
    ):
        super().__init__()

        self.sigma_emb = FourierEncoding(cond_dim, w_min_sigma, w_max_sigma)
        self.diff_emb = FourierEncoding(cond_dim, w_min_sigma, w_max_sigma)

        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim)
        )

    def forward(
        self, sigma_transformed: torch.Tensor, diff: torch.Tensor
    ) -> torch.Tensor:
        cond_emb = self.sigma_emb(sigma_transformed) + self.diff_emb(diff)
        cond_emb = self.cond_mlp(cond_emb)
        return cond_emb
