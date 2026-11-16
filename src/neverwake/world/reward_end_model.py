import torch
import torch.nn as nn


class RewardEndModel(nn.Module):
    """
    Reads the reward and end of a transition off the memory, which has seen the
    frame and the action of that transition (but not the frame it leads to)
    """

    def __init__(self, memory_dim: int):
        super().__init__()

        # Note: no bias on the logits (same as EDELINE)
        self.reward_head = nn.Sequential(
            nn.Linear(memory_dim, memory_dim),
            nn.SiLU(),
            nn.Linear(memory_dim, 3, bias=False),
        )
        self.end_head = nn.Sequential(
            nn.Linear(memory_dim, memory_dim),
            nn.SiLU(),
            nn.Linear(memory_dim, 2, bias=False),
        )

    def forward(self, memory_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        memory_emb: (..., memory_dim)

        Returns the reward logits (..., 3) for [-1, 0, +1] and the end logits (..., 2)
        """
        return self.reward_head(memory_emb), self.end_head(memory_emb)
