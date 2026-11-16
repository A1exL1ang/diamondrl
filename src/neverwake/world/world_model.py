import torch
import torch.nn as nn

from neverwake.world.denoiser import Denoiser
from neverwake.world.memory import MemoryModel
from neverwake.world.reward_end_model import RewardEndModel


class WorldModel(nn.Module):
    """
    The memory reads the frames and actions, the denoiser and the reward / end heads
    both predict from what it gives. All of it is trained by world_model_loss
    """

    def __init__(
        self, memory: MemoryModel, denoiser: Denoiser, reward_end: RewardEndModel
    ):
        super().__init__()
        self.memory = memory
        self.denoiser = denoiser
        self.reward_end = reward_end

        # HarmonyDream: each one is the log of what its loss is divided by, learned
        # so that neither loss drowns out the other. 0 so we start at a weight of 1
        self.log_scale_reward = nn.Parameter(torch.zeros(()))
        self.log_scale_denoiser = nn.Parameter(torch.zeros(()))
