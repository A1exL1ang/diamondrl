import torch
import torch.nn as nn

from diamondrl.models.common import ResBlockSingleBasic, init_lstm


class ActorCriticModel(nn.Module):
    """
    Sees only 1 frame, LSTM hidden gives information about the previous ones
    """

    def __init__(
        self,
        inp_shape: tuple[int, int, int],
        inp_proj_ch: int,
        channels: list[int],
        downsample_after: list[bool],
        lstm_hidden: int,
        num_actions: int,
        group_size: int,
    ):
        super().__init__()

        if not channels:
            raise ValueError("should have at least 1 res block")

        self.inp_proj = nn.Conv2d(inp_shape[0], inp_proj_ch, 3, padding=1)
        self.blocks = nn.ModuleList()
        self.potential_downsamples = nn.ModuleList()

        final_h, final_w = inp_shape[1:]

        for prev_ch, cur_ch, do_down in zip(
            [inp_proj_ch] + channels[:-1], channels, downsample_after, strict=True
        ):
            self.blocks.append(ResBlockSingleBasic(group_size, prev_ch, cur_ch))
            self.potential_downsamples.append(
                nn.MaxPool2d(2) if do_down else nn.Identity()
            )

            if do_down:
                final_h //= 2
                final_w //= 2

        final_flat = channels[-1] * final_h * final_w

        self.lstm_cell = nn.LSTMCell(final_flat, lstm_hidden)

        self.policy_head = nn.Linear(lstm_hidden, num_actions)
        self.value_head = nn.Linear(lstm_hidden, 1)

        # Init
        init_lstm(self.lstm_cell)

        for head in (self.policy_head, self.value_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(
        self, obs: torch.Tensor, lstm_state: tuple[torch.Tensor, torch.Tensor] | None
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        values is (B,)
        """
        enc = self.inp_proj(obs)

        for block, potential_downsample in zip(
            self.blocks, self.potential_downsamples, strict=True
        ):
            enc = block(enc)
            enc = potential_downsample(enc)

        h_out, c_out = self.lstm_cell(enc.flatten(start_dim=1), lstm_state)

        policies = self.policy_head(h_out)
        values = self.value_head(h_out)

        return policies, values.squeeze(1), (h_out, c_out)
