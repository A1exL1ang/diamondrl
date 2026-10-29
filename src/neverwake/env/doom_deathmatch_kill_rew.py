import os
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import vizdoom as vzd

from neverwake.config.env import DoomDeathmatchKillRewEnvConfig
from neverwake.env.base import AbstractEnv


class DoomDeathmatchKillRewEnv(AbstractEnv):
    """
    ViZDoom deathmatch where an episode is one life (death or timeout)
    """

    def __init__(self, cfg: DoomDeathmatchKillRewEnvConfig) -> None:
        self.cfg = cfg

        self.game = vzd.DoomGame()
        self.game.load_config(os.path.join(vzd.scenarios_path, "deathmatch.cfg"))
        self.game.set_window_visible(False)
        self.game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
        self.game.set_screen_format(vzd.ScreenFormat.RGB24)

        # Note: the scenario hides the HUD, so damage and pickup flashes are the only
        # on-screen sign of health and ammo changes
        self.game.set_render_screen_flashes(True)

        # Note: movement, turning and shooting instead of the scenario's 17 binary and
        # 3 delta buttons, so there are no weapon selection buttons
        self.game.set_available_buttons(
            [
                vzd.Button.ATTACK,
                vzd.Button.MOVE_FORWARD,
                vzd.Button.MOVE_BACKWARD,
                vzd.Button.TURN_LEFT,
                vzd.Button.TURN_RIGHT,
                vzd.Button.MOVE_LEFT,
                vzd.Button.MOVE_RIGHT,
            ]
        )
        self.game.init()

        # One action per button combination, in the order of the buttons above
        self.combos = [
            [1, 0, 0, 0, 0, 0, 0],  # Attack
            [0, 1, 0, 0, 0, 0, 0],  # Forward
            [0, 0, 1, 0, 0, 0, 0],  # Backward
            [0, 0, 0, 1, 0, 0, 0],  # Turn left
            [0, 0, 0, 0, 1, 0, 0],  # Turn right
            [0, 0, 0, 0, 0, 1, 0],  # Strafe left
            [0, 0, 0, 0, 0, 0, 1],  # Strafe right
            [0, 1, 0, 1, 0, 0, 0],  # Forward and turn left
            [0, 1, 0, 0, 1, 0, 0],  # Forward and turn right
        ]

        # State
        self.screen = np.zeros((120, 160, 3), dtype=np.uint8)  # Last rendered frame
        self.kills = 0.0

    @property
    def all_actions(self) -> torch.Tensor:
        return torch.arange(len(self.combos))

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
        buttons = self.combos[action]

        # Note: all frameskip tics in one call, which renders only the last one (tic by
        # tic is about 4x slower) and is enough since the kill count only goes up
        self.game.make_action(buttons, self.cfg.frameskip)
        killed = False
        if not self.game.is_episode_finished():
            kills = self.game.get_game_variable(vzd.GameVariable.KILLCOUNT)
            killed = kills > self.kills
            self.kills = kills
            self.screen = self.game.get_state().screen_buffer

        # +1 for killing a monster, death overrides with -1
        terminated = self.game.is_player_dead()
        truncated = self.game.is_episode_finished() and not terminated
        reward = -1.0 if terminated else float(killed)

        # Every episode is a single life so it always ends the game
        info: dict[str, Any] = {"game_end": terminated or truncated}

        # Note: a finished episode has no state, so the last frame repeats
        return self.proc_obs(self.screen), reward, terminated, truncated, info

    def reset(self, seed: int | None) -> tuple[torch.Tensor, dict[str, Any]]:
        # Note: the seed drives the monster spawns, so unseeded resets continue its
        # stream
        if seed is not None:
            self.game.set_seed(seed)
        self.game.new_episode()

        self.kills = self.game.get_game_variable(vzd.GameVariable.KILLCOUNT)
        self.screen = self.game.get_state().screen_buffer
        return self.proc_obs(self.screen), {}

    def render(self) -> np.ndarray:
        return self.screen.copy()

    def close(self) -> None:
        self.game.close()
