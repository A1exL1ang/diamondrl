import torch
import torch.nn as nn

from neverwake.modules.blocks import ResBlockSingleBasic, init_lstm


class ActorCriticModel(nn.Module):
    """
    Sees only 1 frame, LSTM hidden gives information about the previous ones
    """

    ret_low: torch.Tensor
    ret_high: torch.Tensor

    def __init__(
        self,
        inp_shape: tuple[int, int, int],
        inp_proj_ch: int,
        channels: tuple[int, ...],
        downsample_after: tuple[bool, ...],
        lstm_dim: int,
        num_actions: int,
        group_size: int,
        return_percentiles: tuple[float, float],
        return_range_decay: float,
        return_range_min: float,
    ):
        """
        return_percentiles: the low and high end of the running range of the returns
        return_range_decay: how slowly the range follows new batches
        return_range_min: a smaller range counts as this (has to be above 0)
        """
        super().__init__()

        if not channels:
            raise ValueError("should have at least 1 res block")
        if return_range_min <= 0.0:
            raise ValueError("the smallest return range has to be above 0")

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

        self.lstm_cell = nn.LSTMCell(final_flat, lstm_dim)

        self.policy_head = nn.Linear(lstm_dim, num_actions)
        self.value_head = nn.Linear(lstm_dim, 1)

        # The critic works in units of the running range of the imagined returns (see
        # to_return), so it never needs a large output whatever the size of the
        # rewards. The range is the return normalisation of DreamerV3 (Hafner et al.
        # 2023, "Mastering Diverse Domains through World Models", arXiv:2301.04104,
        # section "Actor learning"): the gap between a low and a high percentile of
        # the returns, as a running average over batches, never taken below a floor
        # (the 5th and 95th percentile, a decay of 0.99 and a floor of 1 in that
        # paper). DreamerV3 divides only the advantages by it. Measuring the critic's
        # target in the same units is not DreamerV3's: it is the adaptive target
        # normalisation of van Hasselt et al. 2016 ("Learning values across many
        # orders of magnitude", arXiv:1602.07714), without that paper's step that
        # keeps the critic's outputs unchanged when the range moves
        self.return_percentiles = return_percentiles
        self.return_range_decay = return_range_decay
        self.return_range_min = return_range_min

        # The running low and high percentile of the returns. Only update_return_range
        # moves them (buffers, so they are saved with the weights)
        self.register_buffer("ret_low", torch.zeros(()))
        self.register_buffer("ret_high", torch.zeros(()))

        # Init
        # Note: unlike DIAMOND the heads keep their random init (like IRIS), so a low
        # collection temperature commits to actions and the policy keeps moving
        # before any reward is seen (needed for Freeway)
        init_lstm(self.lstm_cell)

    def forward(
        self, obs: torch.Tensor, lstm_state: tuple[torch.Tensor, torch.Tensor] | None
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        values is (B,) in units of the running range of the returns, see to_return
        """
        enc = self.inp_proj(obs)

        for block, potential_downsample in zip(
            self.blocks, self.potential_downsamples, strict=True
        ):
            enc = block(enc)
            enc = potential_downsample(enc)

        h_out, c_out = self.lstm_cell(enc.flatten(start_dim=1), lstm_state)

        policy_logits = self.policy_head(h_out)
        values = self.value_head(h_out)

        return policy_logits, values.squeeze(1), (h_out, c_out)

    def return_range(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the low end of the running range of the returns and its size. The size
        is never below return_range_min (with 1, small returns are left as they are
        and only large ones are scaled down)
        """
        scale = (self.ret_high - self.ret_low).clamp(min=self.return_range_min)
        return self.ret_low.clone(), scale

    def to_return(self, values: torch.Tensor) -> torch.Tensor:
        """
        values: (...) as the critic outputs them

        Returns them as actual returns (...): the low end of the running range plus
        the output times the range
        """
        low, scale = self.return_range()
        return low + scale * values

    @torch.no_grad()
    def update_return_range(
        self, lamb_ret: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Moves the running percentiles of the returns towards those of this batch

        lamb_ret: (N,) the returns of 1 batch

        Returns the range after the move (see return_range)
        """
        percentiles = torch.tensor(self.return_percentiles, device=lamb_ret.device)
        low, high = torch.quantile(lamb_ret, percentiles)

        decay = self.return_range_decay
        self.ret_low.mul_(decay).add_((1.0 - decay) * low)
        self.ret_high.mul_(decay).add_((1.0 - decay) * high)

        return self.return_range()
