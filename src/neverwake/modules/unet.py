import torch
import torch.nn as nn
import torch.nn.functional as F

from neverwake.modules.blocks import ResBlockCond


class DownBlock(nn.Module):
    def __init__(
        self,
        group_size: int,
        in_ch: int,
        block_ch: int,
        cond_dim: int,
        down_sample: bool,
    ):
        super().__init__()

        self.res_blocks = nn.ModuleList(
            [
                ResBlockCond(group_size, in_ch, block_ch, cond_dim),
                ResBlockCond(group_size, block_ch, block_ch, cond_dim),
            ]
        )
        # (h // 2, w // 2)
        self.down_sample = (
            nn.Conv2d(block_ch, block_ch, kernel_size=3, stride=2, padding=1)
            if down_sample
            else nn.Identity()
        )

    def forward(
        self, x: torch.Tensor, cond_emb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for block in self.res_blocks:
            x = block(x, cond_emb)
        return self.down_sample(x), x


class UpBlock(nn.Module):
    def __init__(self, group_size: int, in_ch: int, block_ch: int, cond_dim: int):
        super().__init__()

        # Conv only applies to the non-skip part interpolation
        self.conv = nn.Conv2d(
            in_ch - block_ch, in_ch - block_ch, kernel_size=3, padding=1
        )

        self.res_blocks = nn.ModuleList(
            [
                ResBlockCond(group_size, in_ch, block_ch, cond_dim),
                ResBlockCond(group_size, block_ch, block_ch, cond_dim),
            ]
        )

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor, cond_emb: torch.Tensor
    ) -> torch.Tensor:
        x = self.conv(F.interpolate(x, size=skip.shape[-2:], mode="nearest"))
        x = torch.cat([x, skip], dim=1)

        for block in self.res_blocks:
            x = block(x, cond_emb)
        return x


class Unet(nn.Module):
    def __init__(
        self,
        inp_ch: int,
        inp_proj_ch: int,
        midcoder_ch: int,
        enc_dec_channels: tuple[int, ...],
        out_ch: int,
        group_size: int,
        cond_dim: int,
        num_heads: int | None,
    ):
        super().__init__()

        if len(enc_dec_channels) < 1:
            raise ValueError("should have at least 1 encoder & decoder")

        # Increase number of channels
        self.inp_proj = nn.Conv2d(inp_ch, inp_proj_ch, 3, padding=1)

        # Encoders and decoders
        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        for i in range(len(enc_dec_channels)):
            in_ch = enc_dec_channels[i - 1] if i else inp_proj_ch
            block_ch = enc_dec_channels[i]
            up_in_ch = block_ch + (
                enc_dec_channels[i + 1]
                if i + 1 < len(enc_dec_channels)
                else midcoder_ch
            )

            # Note: don't sample on the last one
            down_sample = i + 1 < len(enc_dec_channels)

            self.down_blocks.append(
                DownBlock(group_size, in_ch, block_ch, cond_dim, down_sample)
            )
            self.up_blocks.append(UpBlock(group_size, up_in_ch, block_ch, cond_dim))

        # Midcoder with self-attention
        self.midcoder = nn.ModuleList(
            [
                ResBlockCond(
                    group_size, enc_dec_channels[-1], midcoder_ch, cond_dim, num_heads
                ),
                ResBlockCond(group_size, midcoder_ch, midcoder_ch, cond_dim, num_heads),
            ]
        )

        # Output
        self.out_gn = nn.GroupNorm(
            enc_dec_channels[0] // group_size, enc_dec_channels[0]
        )
        self.out_conv = nn.Conv2d(enc_dec_channels[0], out_ch, 3, padding=1)

        assert self.out_conv.bias is not None
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        x = self.inp_proj(x)
        skips: list[torch.Tensor] = []

        for block in self.down_blocks:
            # Note that skip is the result prior to downsampling (if it is done)
            x, skip = block(x, cond_emb)
            skips.append(skip)

        for block in self.midcoder:
            x = block(x, cond_emb)

        for block, skip in reversed(list(zip(self.up_blocks, skips, strict=True))):
            x = block(x, skip, cond_emb)

        return self.out_conv(F.silu(self.out_gn(x)))
