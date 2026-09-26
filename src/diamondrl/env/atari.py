from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3.common.atari_wrappers import (
    EpisodicLifeEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

from diamondrl.config.env import AtariEnvConfig
from diamondrl.env.base import AbstractEnv


class AtariEnv(AbstractEnv):
    def __init__(self, cfg: AtariEnvConfig) -> None:
        self.cfg = cfg

        self.env = gym.make(
            id=f"ale_py:ALE/{cfg.game}-v5",
            frameskip=1,
            render_mode="rgb_array",
            full_action_space=False,
            repeat_action_probability=0.0,
        )
        self.env = NoopResetEnv(self.env, cfg.max_noop)

        # Note that we use this frameskip to avoid sprite flicker
        self.env = MaxAndSkipEnv(self.env, cfg.frameskip)

        self.env = EpisodicLifeEnv(self.env)

    @property
    def all_actions(self) -> torch.Tensor:
        return torch.arange(self.env.action_space.n)  # type: ignore[attr-defined]

    @property
    def obs_shape(self) -> tuple[int, int, int]:
        return self.cfg.rescaled_shape

    def _proc_obs(self, original_obs: np.ndarray) -> torch.Tensor:
        # (h, w, c) --> (c, h, w) --> (c, new_h, new_w)
        obs = torch.tensor(original_obs).permute(2, 0, 1).float()
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
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["game_end"] = self.env.was_real_done  # type: ignore
        return self._proc_obs(obs), float(reward), terminated, truncated, info

    def reset(self, seed: int | None) -> tuple[torch.Tensor, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed)
        return self._proc_obs(obs), info

    def render(self) -> np.ndarray:
        return np.array(self.env.render())

    def close(self) -> None:
        self.env.close()
