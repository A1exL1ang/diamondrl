import torch
import torch.nn as nn

from neverwake.modules.twohot import twohot_bins


class RewardEndModel(nn.Module):
    """
    Reads the reward and end of a transition off the memory, which has seen the
    frame and the action of that transition (but not the frame it leads to)
    """

    bins: torch.Tensor

    def __init__(self, memory_dim: int, num_bins: int, bin_limit: float):
        super().__init__()

        # Note: no bias on the logits (same as EDELINE)
        reward_hidden = nn.Linear(memory_dim, memory_dim)
        reward_out = nn.Linear(memory_dim, num_bins, bias=False)
        self.reward_head = nn.Sequential(reward_hidden, nn.SiLU(), reward_out)
        self.end_head = nn.Sequential(
            nn.Linear(memory_dim, memory_dim),
            nn.SiLU(),
            nn.Linear(memory_dim, 2, bias=False),
        )

        # The values of the twohot bins that the reward logits are over (they come
        # from the config, so they are not saved with the weights)
        self.register_buffer("bins", twohot_bins(num_bins, bin_limit), persistent=False)

        # Note: zero init (same as DreamerV3) so the predicted reward starts at
        # exactly 0, a random init would start it anywhere in the range of the bins
        nn.init.zeros_(reward_out.weight)

    def forward(self, memory_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        memory_emb: (..., memory_dim)

        Returns the reward logits (..., num_bins) over the twohot bins and the end
        logits (..., 2)
        """
        return self.reward_head(memory_emb), self.end_head(memory_emb)
