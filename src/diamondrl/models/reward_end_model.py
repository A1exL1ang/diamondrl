import torch
import torch.nn as nn

from diamondrl.models.common import ResBlockCond, init_lstm


class RewardEndModel(nn.Module):
    def __init__(
        self,
        inp_shape: tuple[int, int, int],  # Stack obs and next_obs
        inp_proj_ch: int,
        channels: list[int],
        downsample_after: list[bool],
        lstm_hidden: int,
        group_size: int,
        num_actions: int,
        cond_dim: int,
    ):
        super().__init__()

        self.action_emb = nn.Embedding(num_actions, cond_dim)
        self.inp_proj = nn.Conv2d(inp_shape[0], inp_proj_ch, 3, padding=1)

        self.blocks = nn.ModuleList()
        self.potential_downsamples = nn.ModuleList()

        final_h, final_w = inp_shape[1:]

        for prev_ch, cur_ch, do_down in zip(
            [inp_proj_ch] + channels[:-1], channels, downsample_after, strict=True
        ):
            self.blocks.append(ResBlockCond(group_size, prev_ch, cur_ch, cond_dim))
            self.potential_downsamples.append(
                nn.MaxPool2d(2) if do_down else nn.Identity()
            )

            if do_down:
                final_h //= 2
                final_w //= 2

        final_flat = channels[-1] * final_h * final_w
        self.lstm = nn.LSTM(final_flat, lstm_hidden, batch_first=True)

        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden), nn.SiLU(), nn.Linear(lstm_hidden, 5)
        )

        # Init, no benefit in init head
        init_lstm(self.lstm)

    def forward(
        self,
        obs_ctx_stk: torch.Tensor,
        action_seq: torch.Tensor,
        next_obs_stk: torch.Tensor,
        lstm_state: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        obs_ctx_stk: (B, T, C, H, W)
        action_seq: (B, T)
        next_obs_seq: (B, T, C, H, W)
        lstm_state: ((1, B, dim), (1, B, dim)) where the 1 denotes the layer
                    Note: batch first only applies to input/output and not hidden
        """
        batch, seq = obs_ctx_stk.shape[:2]
        obs_ctx_stk = obs_ctx_stk.flatten(start_dim=0, end_dim=1)
        action_seq = action_seq.flatten(start_dim=0, end_dim=1)
        next_obs_stk = next_obs_stk.flatten(start_dim=0, end_dim=1)

        enc = torch.cat([obs_ctx_stk, next_obs_stk], dim=1)
        cond_emb = self.action_emb(action_seq)

        enc = self.inp_proj(enc)

        for block, potential_downsample in zip(
            self.blocks, self.potential_downsamples, strict=True
        ):
            enc = block(enc, cond_emb)
            enc = potential_downsample(enc)

        # enc is now (B * T, C, H, W) and we need to turn it into (B, T, dim)
        enc = enc.reshape(batch, seq, -1)
        out, next_hidden = self.lstm(enc, lstm_state)

        result = self.head(out)

        # (B, T, logit_dim)
        return result[..., :3], result[..., 3:], next_hidden
