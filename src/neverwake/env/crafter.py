import math
from typing import Any

import crafter  # type: ignore[import-untyped]
import numpy as np
import torch

from neverwake.config.env import CrafterEnvConfig
from neverwake.env.base import AbstractEnv


class CrafterEnv(AbstractEnv):
    """
    The original Crafter from pixels with its own rewards (+1 for every achievement
    unlocked for the first time in an episode, +-0.1 for every point of health gained
    or lost), where an episode is one life (death or the step limit)
    """

    def __init__(self, cfg: CrafterEnvConfig) -> None:
        if cfg.rescaled_shape[0] != 3:
            raise ValueError(
                f"rescaled_shape {cfg.rescaled_shape} should have 3 channels"
            )
        self.cfg = cfg

        # Note: no frameskip since every action moves at most one tile
        self.env = self.make_env(seed=0)
        self.num_actions = int(self.env.action_space.n)

        # State
        _, h, w = cfg.rescaled_shape
        self.screen = np.zeros((h, w, 3), dtype=np.uint8)  # Last rendered frame

    def make_env(self, seed: int) -> Any:
        # Note: Crafter draws its frames at the size it is given (width by height),
        # so nothing is resized here
        _, h, w = self.cfg.rescaled_shape
        return crafter.Env(size=(w, h), seed=seed)

    @property
    def all_actions(self) -> torch.Tensor:
        return torch.arange(self.num_actions)

    @property
    def obs_shape(self) -> tuple[int, int, int]:
        return self.cfg.rescaled_shape

    @property
    def achievements(self) -> tuple[str, ...]:
        # Note: the package reads its constants from a data file when it is imported
        return tuple(crafter.constants.achievements)  # type: ignore[attr-defined]

    def achievement_score(self, success_rates: dict[str, float]) -> float:
        # The Crafter score (Hafner 2022, "Benchmarking the Spectrum of Agent
        # Capabilities", arXiv:2109.06780, section 3.3): exp(mean(ln(1 + rate))) - 1
        # over all 22 achievements, a geometric mean so that rare ones count
        logs = [math.log1p(success_rates[name]) for name in self.achievements]
        return math.expm1(sum(logs) / len(logs))

    def proc_obs(self, original_obs: np.ndarray) -> torch.Tensor:
        # (h, w, c) --> (c, h, w)
        obs = torch.tensor(original_obs).permute(2, 0, 1).float()
        assert obs.shape == self.cfg.rescaled_shape

        # We normalize to [-1, +1] to mirror DIAMOND so that we can use their sigma_data
        return (obs - (255 / 2)) / (255 / 2)

    def step(
        self, action: int
    ) -> tuple[torch.Tensor, float, bool, bool, dict[str, Any]]:
        self.screen, reward, done, info = self.env.step(action)

        # Note: the game ends on death (Crafter then reports a discount of 0) or at
        # its step limit
        terminated = info["discount"] == 0.0
        truncated = bool(done) and not terminated

        # Every episode is a single life so it always ends the game
        info["game_end"] = terminated or truncated

        # Crafter counts how often each achievement was reached in this episode
        info["unlocked"] = [
            name for name, count in info["achievements"].items() if count > 0
        ]

        return self.proc_obs(self.screen), float(reward), terminated, truncated, info

    def reset(self, seed: int | None) -> tuple[torch.Tensor, dict[str, Any]]:
        # Note: the seed generates the worlds, so unseeded resets continue its stream
        if seed is not None:
            self.env = self.make_env(seed)
        self.screen = self.env.reset()
        return self.proc_obs(self.screen), {}

    def render(self) -> np.ndarray:
        return self.screen.copy()

    def close(self) -> None:
        # Note: Crafter holds nothing that has to be released
        pass
