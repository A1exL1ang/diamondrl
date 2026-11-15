import torch
import torch.nn as nn
from mambapy.mamba import Mamba, MambaConfig

from neverwake.modules.blocks import ResBlockSingleBasic


class MemoryModel(nn.Module):
    """
    Sees 1 frame and the action taken on it per step, the Mamba state gives
    information about all the previous ones
    """

    def __init__(
        self,
        inp_shape: tuple[int, int, int],
        inp_proj_ch: int,
        channels: tuple[int, ...],
        downsample_after: tuple[bool, ...],
        group_size: int,
        num_actions: int,
        action_dim: int,
        mamba_dim: int,
        mamba_layers: int,
    ):
        super().__init__()

        if not channels:
            raise ValueError("should have at least 1 res block")

        self.inp_proj = nn.Conv2d(inp_shape[0], inp_proj_ch, 3, padding=1)
        self.blocks = nn.ModuleList()
        self.potential_downsamples = nn.ModuleList()

        final_h, final_w = inp_shape[1:]

        for prev_ch, cur_ch, do_down in zip(
            (inp_proj_ch,) + channels[:-1], channels, downsample_after, strict=True
        ):
            self.blocks.append(ResBlockSingleBasic(group_size, prev_ch, cur_ch))
            self.potential_downsamples.append(
                nn.MaxPool2d(2) if do_down else nn.Identity()
            )

            if do_down:
                final_h //= 2
                final_w //= 2

        final_flat = channels[-1] * final_h * final_w

        self.action_emb = nn.Embedding(num_actions, action_dim)
        self.mamba_proj = nn.Sequential(
            nn.Linear(final_flat + action_dim, mamba_dim), nn.SiLU()
        )

        # Note: the rest of the Mamba settings stay at the mambapy defaults
        # (state size 16, expand 2, conv width 4)
        self.mamba = Mamba(MambaConfig(d_model=mamba_dim, n_layers=mamba_layers))
        self.out_norm = nn.RMSNorm(mamba_dim, eps=1e-5)

    def embed(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        obs: (N, C, H, W)
        actions: (N,)

        Returns (N, mamba_dim), what Mamba is fed for each (frame, action) pair
        """
        enc = self.inp_proj(obs)

        for block, potential_downsample in zip(
            self.blocks, self.potential_downsamples, strict=True
        ):
            enc = block(enc)
            enc = potential_downsample(enc)

        enc = torch.cat([enc.flatten(start_dim=1), self.action_emb(actions)], dim=1)
        return self.mamba_proj(enc)

    def forward(self, obs_seq: torch.Tensor, action_seq: torch.Tensor) -> torch.Tensor:
        """
        obs_seq: (B, T, C, H, W)
        action_seq: (B, T) where entry t is the action taken on frame t

        Returns (B, T, mamba_dim) where entry t has seen frames and actions 0..t
        """
        batch, seq = obs_seq.shape[:2]

        emb = self.embed(
            obs_seq.flatten(start_dim=0, end_dim=1),
            action_seq.flatten(start_dim=0, end_dim=1),
        )
        return self.out_norm(self.mamba(emb.reshape(batch, seq, -1)))

    def step(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        obs: (B, C, H, W)
        actions: (B,)
        state: (ssm_state, conv_inputs) or None to start with no history
            ssm_state: (layers, B, d_inner, d_state)
            conv_inputs: (layers, B, d_inner, d_conv - 1) which are the last inputs
                         of each layer's conv over time

        Same as forward but 1 frame at a time: stepping through a sequence from
        None gives the same outputs as forward on the whole sequence
        """
        cfg = self.mamba.config

        if state is None:
            shape = (cfg.n_layers, obs.shape[0], cfg.d_inner)
            state = (
                torch.zeros(*shape, cfg.d_state, device=obs.device),
                torch.zeros(*shape, cfg.d_conv - 1, device=obs.device),
            )

        # mambapy wants a list with one (ssm_state, conv_inputs) pair per layer
        caches = list(zip(state[0].unbind(0), state[1].unbind(0), strict=True))
        out, caches = self.mamba.step(self.embed(obs, actions), caches)

        next_state = (
            torch.stack([ssm_state for ssm_state, _ in caches]),
            torch.stack([conv_inputs for _, conv_inputs in caches]),
        )
        return self.out_norm(out), next_state
