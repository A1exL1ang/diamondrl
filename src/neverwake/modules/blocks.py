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


class SelfAttention2d(nn.Module):
    def __init__(self, ch: int, group_size: int, num_heads: int):
        assert num_heads > 0 and ch % num_heads == 0
        assert group_size > 0 and ch % group_size == 0

        super().__init__()

        self.num_heads = num_heads

        self.norm = nn.GroupNorm(ch // group_size, num_channels=ch)
        self.qkv_proj = nn.Linear(ch, ch * 3)
        self.out_proj = nn.Linear(ch, ch)

        # Note: zero init so the block starts as the identity
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)

        Each pixel is a token of size C, multi-head attention over all pixels
        """
        batch_size, ch, height, width = x.shape

        h = self.norm(x)

        # (B, C, H, W) --> (B, H*W, C)
        h = h.flatten(2).transpose(1, 2)
        q, k, v = torch.chunk(self.qkv_proj(h), chunks=3, dim=2)

        # (B, H*W, C) --> (B, H*W, heads, C / heads) --> (B, heads, H*W, C / heads)
        q, k, v = (
            t.reshape(
                batch_size, height * width, self.num_heads, ch // self.num_heads
            ).transpose(1, 2)
            for t in (q, k, v)
        )

        h = F.scaled_dot_product_attention(q, k, v)

        # (B, heads, H*W, C / heads) --> (B, H*W, C) --> (B, C, H, W)
        h = h.transpose(1, 2).reshape(batch_size, height * width, ch)
        h = self.out_proj(h)
        h = h.transpose(1, 2).reshape(batch_size, ch, height, width)

        return x + h


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
    def __init__(
        self,
        group_size: int,
        in_ch: int,
        out_ch: int,
        cond_dim: int,
        num_heads: int | None = None,
    ):
        super().__init__()

        self.units = nn.ModuleList(
            [
                AdaGNConv(group_size, in_ch, out_ch, cond_dim),
                AdaGNConv(group_size, out_ch, out_ch, cond_dim),
            ]
        )
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

        self.attn = (
            SelfAttention2d(out_ch, group_size, num_heads)
            if num_heads is not None
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        h = x
        for unit in self.units:
            h = unit(h, cond_emb)
        return self.attn(self.skip(x) + h)


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
