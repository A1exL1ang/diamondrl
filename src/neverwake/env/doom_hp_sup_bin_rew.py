import os
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import vizdoom as vzd

from neverwake.config.env import DoomHpSupBinRewEnvConfig
from neverwake.env.base import AbstractEnv


class DoomHpSupBinRewEnv(AbstractEnv):
    """
    ViZDoom health gathering supreme where an episode is one life (death or timeout)
    """

    def __init__(self, cfg: DoomHpSupBinRewEnvConfig) -> None:
        self.cfg = cfg

        # Note: the scenario fixes the buttons (turn left, turn right, forward)
        self.game = vzd.DoomGame()
        self.game.load_config(
            os.path.join(vzd.scenarios_path, "health_gathering_supreme.cfg")
        )
        self.game.set_window_visible(False)
        self.game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
        self.game.set_screen_format(vzd.ScreenFormat.RGB24)
        self.game.init()
        self.num_buttons = len(self.game.get_available_buttons())

        # State
        self.screen = np.zeros((120, 160, 3), dtype=np.uint8)  # Last rendered frame
        self.health = 0.0

    @property
    def all_actions(self) -> torch.Tensor:
        return torch.arange(self.num_buttons)

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

        # Temporarily add and remove batch dimension
        obs = F.interpolate(
            obs.unsqueeze(0), size=self.cfg.rescaled_shape[1:], mode="area"
        ).squeeze(0)

        # We normalize to [-1, +1] to mirror DIAMOND so that we can use their sigma_data
        return (obs - (255 / 2)) / (255 / 2)

    def step(
        self, action: int
    ) -> tuple[torch.Tensor, float, bool, bool, dict[str, Any]]:
        # One button per action, held for frameskip tics
        buttons = [0] * self.num_buttons
        buttons[action] = 1

        # Note: tic by tic since health from a medkit within the skip can be
        # hidden by the acid damage
        picked_up = False
        for _ in range(self.cfg.frameskip):
            self.game.make_action(buttons, 1)
            if self.game.is_episode_finished():
                break
            health = self.game.get_game_variable(vzd.GameVariable.HEALTH)
            picked_up |= health > self.health
            self.health = health
            self.screen = self.game.get_state().screen_buffer

        # +1 for picking up a medkit, death overrides with -1
        terminated = self.game.is_player_dead()
        truncated = self.game.is_episode_finished() and not terminated
        reward = -1.0 if terminated else float(picked_up)

        # Every episode is a single life so it always ends the game
        info: dict[str, Any] = {"game_end": terminated or truncated}

        # Note: a finished episode has no state, so the last frame repeats
        return self.proc_obs(self.screen), reward, terminated, truncated, info

    def reset(self, seed: int | None) -> tuple[torch.Tensor, dict[str, Any]]:
        # Note: the seed places the medkits, so unseeded resets continue its stream
        if seed is not None:
            self.game.set_seed(seed)
        self.game.new_episode()

        self.health = self.game.get_game_variable(vzd.GameVariable.HEALTH)
        self.screen = self.game.get_state().screen_buffer
        return self.proc_obs(self.screen), {}

    def render(self) -> np.ndarray:
        return self.screen.copy()

    def close(self) -> None:
        self.game.close()
