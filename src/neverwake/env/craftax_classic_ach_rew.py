from typing import Any

import jax
import numpy as np
import torch
import torch.nn.functional as F
from craftax.craftax_classic.envs import (  # type: ignore[import-untyped]
    craftax_pixels_env,
)

from neverwake.config.env import CraftaxClassicAchRewEnvConfig
from neverwake.env.base import AbstractEnv


class CraftaxClassicAchRewEnv(AbstractEnv):
    """
    Craftax-Classic (Crafter in JAX) from pixels where an episode is one life (death or
    the step limit)
    """

    def __init__(self, cfg: CraftaxClassicAchRewEnvConfig) -> None:
        # Note: the view is 9 by 9 tiles of 7 pixels (7 rows of map, 2 of inventory)
        if cfg.rescaled_shape[0] != 3 or min(cfg.rescaled_shape[1:]) < 63:
            raise ValueError(
                f"rescaled_shape {cfg.rescaled_shape} should be 3 channels of at least "
                "63 by 63 pixels"
            )
        self.cfg = cfg

        # Note: no frameskip since every action moves at most one tile
        self.env = craftax_pixels_env.CraftaxClassicPixelsEnvNoAutoReset()
        self.params = self.env.default_params
        self.env_step = jax.jit(self.env.step)
        self.env_reset = jax.jit(self.env.reset)
        self.num_actions = int(self.env.action_space(self.params).n)

        # State
        self.key = jax.random.PRNGKey(0)
        self.state: Any = None
        self.screen = np.zeros((63, 63, 3), dtype=np.uint8)  # Last rendered frame

    @property
    def all_actions(self) -> torch.Tensor:
        return torch.arange(self.num_actions)

    @property
    def obs_shape(self) -> tuple[int, int, int]:
        return self.cfg.rescaled_shape

    def proc_obs(self, original_obs: np.ndarray) -> torch.Tensor:
        # (h, w, c) --> (c, h, w) --> (c, new_h, new_w)
        obs = torch.tensor(original_obs).permute(2, 0, 1).float()
        assert (
            len(obs.shape) == len(self.cfg.rescaled_shape) == 3
            and obs.shape[0] == self.cfg.rescaled_shape[0]
        )

        # Note: pad with black instead of resizing, so tiles and inventory digits stay
        # pixel-exact
        _, h, w = obs.shape
        obs = F.pad(
            obs, (0, self.cfg.rescaled_shape[2] - w, 0, self.cfg.rescaled_shape[1] - h)
        )

        # We normalize to [-1, +1] to mirror DIAMOND so that we can use their sigma_data
        return (obs - (255 / 2)) / (255 / 2)

    def to_screen(self, obs: jax.Array) -> np.ndarray:
        # (h, w, c) in [0, 1] --> uint8 (h, w, c)
        return (np.asarray(obs) * 255).round().astype(np.uint8)

    def step(
        self, action: int
    ) -> tuple[torch.Tensor, float, bool, bool, dict[str, Any]]:
        self.key, key = jax.random.split(self.key)
        achievements = int(self.state.achievements.sum())
        obs, self.state, _, done, _ = self.env_step(
            key, self.state, action, self.params
        )
        self.screen = self.to_screen(obs)

        # Note: the game also ends on death or walking into lava
        truncated = bool(self.state.timestep >= self.params.max_timesteps)
        terminated = bool(done) and not truncated

        # +1 for unlocking any new achievement, death overrides with -1
        unlocked = int(self.state.achievements.sum()) > achievements
        reward = -1.0 if terminated else float(unlocked)

        # Every episode is a single life so it always ends the game
        info: dict[str, Any] = {"game_end": terminated or truncated}

        return self.proc_obs(self.screen), reward, terminated, truncated, info

    def reset(self, seed: int | None) -> tuple[torch.Tensor, dict[str, Any]]:
        # Note: the seed generates the world, so unseeded resets continue its stream
        if seed is not None:
            self.key = jax.random.PRNGKey(seed)
        self.key, key = jax.random.split(self.key)
        obs, self.state = self.env_reset(key, self.params)
        self.screen = self.to_screen(obs)
        return self.proc_obs(self.screen), {}

    def render(self) -> np.ndarray:
        return self.screen.copy()

    def close(self) -> None:
        # Note: a pure JAX env has nothing to release
        pass
