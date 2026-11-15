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

        # Note: orthogonal init so downsample won't destroy the signal at init
        if isinstance(self.down_sample, nn.Conv2d):
            nn.init.orthogonal_(self.down_sample.weight)

    def forward(
        self, x: torch.Tensor, cond_emb: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        # One skip per tensor at this resolution: input, then each res block output
        skips = [x]
        for block in self.res_blocks:
            x = block(x, cond_emb)
            skips.append(x)
        return self.down_sample(x), skips


class UpBlock(nn.Module):
    def __init__(
        self,
        group_size: int,
        below_ch: int,
        block_ch: int,
        out_ch: int,
        cond_dim: int,
        up_sample: bool,
    ):
        """
        Mirror of a DownBlock(in_ch, block_ch): takes below_ch channels from the
        level below, works at block_ch and gives back out_ch = the DownBlock's in_ch
        """
        super().__init__()

        self.up_sample = up_sample

        # Refines interpolated part if we interpolate
        self.up_conv = (
            nn.Conv2d(below_ch, below_ch, kernel_size=3, padding=1)
            if up_sample
            else nn.Identity()
        )

        # Note that we have 3 res-blocks this time with each ones inputs taking in
        # a skip
        self.res_blocks = nn.ModuleList(
            [
                # + 2nd DownBlock res block output (block_ch)
                ResBlockCond(group_size, below_ch + block_ch, block_ch, cond_dim),
                # + 1st DownBlock res block output (block_ch)
                ResBlockCond(group_size, block_ch + block_ch, block_ch, cond_dim),
                # + the DownBlock's input (out_ch)
                ResBlockCond(group_size, block_ch + out_ch, out_ch, cond_dim),
            ]
        )

    def forward(
        self, x: torch.Tensor, skips: list[torch.Tensor], cond_emb: torch.Tensor
    ) -> torch.Tensor:
        if self.up_sample:
            x = F.interpolate(x, size=skips[-1].shape[-2:], mode="nearest")
        x = self.up_conv(x)

        for block, skip in zip(self.res_blocks, reversed(skips), strict=True):
            x = block(torch.cat([x, skip], dim=1), cond_emb)
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

            # Note: don't down/up-sample on the ones adjacent to the midcoder
            is_deepest = i + 1 == len(enc_dec_channels)
            below_ch = midcoder_ch if is_deepest else block_ch

            self.down_blocks.append(
                DownBlock(group_size, in_ch, block_ch, cond_dim, not is_deepest)
            )
            # The UpBlock undoes the DownBlock so it gives back in_ch channels
            self.up_blocks.append(
                UpBlock(group_size, below_ch, block_ch, in_ch, cond_dim, not is_deepest)
            )

        # Midcoder with self-attention
        self.midcoder = nn.ModuleList(
            [
                ResBlockCond(
                    group_size, enc_dec_channels[-1], midcoder_ch, cond_dim, num_heads
                ),
                ResBlockCond(group_size, midcoder_ch, midcoder_ch, cond_dim, num_heads),
            ]
        )

        # Output (the top UpBlock gives back inp_proj_ch channels)
        self.out_gn = nn.GroupNorm(inp_proj_ch // group_size, inp_proj_ch)
        self.out_conv = nn.Conv2d(inp_proj_ch, out_ch, 3, padding=1)

        assert self.out_conv.bias is not None
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        x = self.inp_proj(x)
        all_skips: list[list[torch.Tensor]] = []

        for block in self.down_blocks:
            # Note that the skips are the results prior to downsampling (if it is done)
            x, skips = block(x, cond_emb)
            all_skips.append(skips)

        for block in self.midcoder:
            x = block(x, cond_emb)

        for block, skips in reversed(list(zip(self.up_blocks, all_skips, strict=True))):
            x = block(x, skips, cond_emb)

        return self.out_conv(F.silu(self.out_gn(x)))
