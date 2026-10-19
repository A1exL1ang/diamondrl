from typing import Any

import gym_super_mario_bros  # type: ignore[import-untyped]
import numpy as np
import torch
import torch.nn.functional as F
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT  # type: ignore[import-untyped]
from nes_py.wrappers import JoypadSpace  # type: ignore[import-untyped]

from neverwake.config.env import MarioForwardRewL1EnvConfig
from neverwake.env.base import AbstractEnv


class MarioForwardRewL1Env(AbstractEnv):
    """
    Super Mario Bros level 1-1 where an episode is one life (death or the flag)
    """

    def __init__(self, cfg: MarioForwardRewL1EnvConfig) -> None:
        # Note: Mario's top speed is 3 pixels per frame, so with a bigger block a step
        # crosses at most one boundary and rewards stay in {-1, 0, +1}
        if cfg.block_size <= 3 * cfg.frameskip:
            raise ValueError(
                f"block_size {cfg.block_size} should exceed Mario's top speed of "
                f"{3 * cfg.frameskip} pixels per step"
            )
        self.cfg = cfg

        # Note: only level 1-1, where the actions can't enter pipes or warp
        self.env = JoypadSpace(
            gym_super_mario_bros.make("SuperMarioBros-1-1-v0", render_mode="rgb_array"),
            SIMPLE_MOVEMENT,
        )

        # State
        self.x = 0  # Mario's horizontal position in the level (pixels)

    @property
    def all_actions(self) -> torch.Tensor:
        return torch.arange(len(SIMPLE_MOVEMENT))

    @property
    def obs_shape(self) -> tuple[int, int, int]:
        return self.cfg.rescaled_shape

    def proc_obs(self, original_obs: np.ndarray) -> torch.Tensor:
        # (h, w, c) --> (c, h, w) --> (c, new_h, new_w)
        # Note: copy since nes-py's screen is a view with negative strides
        obs = torch.tensor(original_obs.copy()).permute(2, 0, 1).float()
        assert (
            len(obs.shape) == len(self.cfg.rescaled_shape) == 3
            and obs.shape[0] == self.cfg.rescaled_shape[0]
        )

        # Temporarily add and remove batch dimension
        obs = F.interpolate(
            obs.unsqueeze(0), size=self.cfg.rescaled_shape[1:], mode="area"
        ).squeeze(0)

        # We normalize to [-1, +1] to mirror DIAMOND so that we can use their sigma_data
        return (obs - (255 / 2)) / (255 / 2)

    def step(
        self, action: int
    ) -> tuple[torch.Tensor, float, bool, bool, dict[str, Any]]:
        obs, _, terminated, truncated, info = self.env.step(action)
        for _ in range(self.cfg.frameskip - 1):
            if terminated or truncated:
                break
            obs, _, terminated, truncated, info = self.env.step(action)

        # +1/-1 for crossing a block boundary to the right/left, death and flag override
        x = int(info["x_pos"])
        if info["flag_get"]:
            reward = 1.0
        elif terminated:
            reward = -1.0
        elif abs(x - self.x) < self.cfg.block_size:
            reward = float(x // self.cfg.block_size - self.x // self.cfg.block_size)
        else:
            reward = 0.0  # Note: jumps bigger than a block are teleports (e.g. pipes)
        self.x = x

        # Every episode is a single life so it always ends the game
        info["game_end"] = bool(terminated or truncated)

        # Note: only the last frame since a max over the last 2 (like MaxAndSkipEnv)
        # ghosts the scrolling background, and Super Mario Bros barely flickers
        return self.proc_obs(obs), reward, bool(terminated), bool(truncated), info

    def reset(self, seed: int | None) -> tuple[torch.Tensor, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed)

        # Note: the level always starts the same way, so a random number of no-op
        # frames gives varied starts (like NoopResetEnv)
        num_noops = int(self.env.unwrapped.np_random.integers(self.cfg.max_noop + 1))
        for _ in range(num_noops):
            obs, _, terminated, truncated, info = self.env.step(0)
            if terminated or truncated:
                obs, info = self.env.reset()

        self.x = int(info["x_pos"])
        return self.proc_obs(obs), info

    def render(self) -> np.ndarray:
        return np.array(self.env.render())

    def close(self) -> None:
        self.env.close()
