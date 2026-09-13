import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaGNConv(nn.Module):
    def __init__(self, group_size: int, in_ch: int, out_ch: int, cond_dim: int):
        super().__init__()
        self.norm = nn.GroupNorm(in_ch // group_size, in_ch, affine=False)
        self.ada = nn.Linear(cond_dim, 2 * in_ch)
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)

    def forward(self, x: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)  # (B, C, H, W)
        scale, shift = self.ada(cond_emb).chunk(2, dim=-1)  # (B, C)

        # Note: input is centered at 0 so add 1 to scale for initialization reasons
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]

        h = F.silu(h)
        return self.conv(h)


class ResBlockSingleBasic(nn.Module):
    def __init__(self, group_size: int, in_ch: int, out_ch: int):
        super().__init__()
        self.norm = nn.GroupNorm(in_ch // group_size, in_ch)
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(F.silu(self.norm(x)))
        return self.skip(x) + h


class ResBlockCond(nn.Module):
    def __init__(self, group_size: int, in_ch: int, out_ch: int, cond_dim: int):
        super().__init__()

        self.units = nn.ModuleList(
            [
                AdaGNConv(group_size, in_ch, out_ch, cond_dim),
                AdaGNConv(group_size, out_ch, out_ch, cond_dim),
            ]
        )
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        h = x
        for unit in self.units:
            h = unit(h, cond_emb)
        return self.skip(x) + h


def init_lstm(lstm: nn.Module) -> None:
    """
    Standard better LSTM initialization.

    weight_ih: mapping input to hidden and since LSTM doesnt use ReLU
    weight_hh: recurrent weight which we set to be orthogonal
    bias: generally set to 0 but
    """
    for name, param in lstm.named_parameters():
        if "weight_ih" in name:
            for gate in param.chunk(4, dim=0):
                nn.init.xavier_uniform_(gate)
        elif "weight_hh" in name:
            for gate in param.chunk(4, dim=0):
                nn.init.orthogonal_(gate)

        elif "bias" in name:
            nn.init.zeros_(param)

            if "bias_ih" in name:
                # Make forget-gate bias 1
                n = param.shape[0]
                nn.init.constant_(param[n // 4 : n // 2], 1.0)
