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

        # Note: zero init so the block starts as the identity (like DIAMOND)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)

        Keeps number of channels is size of the token

        Applies group norm
        Then does MHA where a token represents a pixel and walks along the channel
        """
        batch_size, ch, height, width = x.shape

        h = self.norm(x)

        # (B, C, H*W) --> (B, H*W, C) where dim=1 denotes the token
        h = h.flatten(2).transpose(1, 2)
        q, k, v = torch.chunk(self.qkv_proj(h), chunks=3, dim=2)

        # Prep for the MHA
        # (B, H*W, C) --> (B, H*W, heads, C / heads) --> (B, heads, H*W, C / heads)
        q = q.reshape(
            batch_size, height * width, self.num_heads, ch // self.num_heads
        ).transpose(1, 2)

        k = k.reshape(
            batch_size, height * width, self.num_heads, ch // self.num_heads
        ).transpose(1, 2)

        v = v.reshape(
            batch_size, height * width, self.num_heads, ch // self.num_heads
        ).transpose(1, 2)

        # (B, heads, H*W, C / heads)
        h = F.scaled_dot_product_attention(q, k, v)

        # (B, heads, H*W, C / heads) --> (B, H*W, heads, C / heads) --> (B, H*W, C)
        h = h.transpose(1, 2).reshape(batch_size, height * width, ch)
        h = self.out_proj(h)

        # (B, H*W, C) --> (B, C, H*W) --> (B, C, H, W)
        h = h.transpose(1, 2).reshape(batch_size, ch, height, width)

        return x + h


class ResBlockCond(nn.Module):
    def __init__(
        self,
        group_size: int,
        in_ch: int,
        out_ch: int,
        cond_dim: int,
        num_heads: int | None,
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
            SelfAttention2d(ch=out_ch, group_size=group_size, num_heads=num_heads)
            if num_heads is not None
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        h = x
        for unit in self.units:
            h = unit(h, cond_emb)
        return self.attn(self.skip(x) + h)


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
                ResBlockCond(group_size, in_ch, block_ch, cond_dim, num_heads=None),
                ResBlockCond(group_size, block_ch, block_ch, cond_dim, num_heads=None),
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
                ResBlockCond(group_size, in_ch, block_ch, cond_dim, num_heads=None),
                ResBlockCond(group_size, block_ch, block_ch, cond_dim, num_heads=None),
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

            # Note: the last level doesn't downsample so the midcoder runs at the same
            # resolution as it (like DIAMOND)
            # TODO: make this change reflected in DIAMOND

            down_sample = i + 1 < len(enc_dec_channels)

            self.down_blocks.append(
                DownBlock(group_size, in_ch, block_ch, cond_dim, down_sample)
            )
            self.up_blocks.append(UpBlock(group_size, up_in_ch, block_ch, cond_dim))

        # Midcoder
        self.midcoder = nn.ModuleList(
            [
                ResBlockCond(
                    group_size,
                    enc_dec_channels[-1],
                    midcoder_ch,
                    cond_dim,
                    num_heads=num_heads,
                ),
                ResBlockCond(
                    group_size, midcoder_ch, midcoder_ch, cond_dim, num_heads=num_heads
                ),
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
