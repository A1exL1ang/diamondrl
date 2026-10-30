import struct
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from neverwake.config.env import ProcgenEnvConfig
from neverwake.env.base import AbstractEnv


class ProcgenEnv(AbstractEnv):
    """
    A Procgen game (e.g. starpilot) stepped in a Python 3.10 subprocess, since procgen's
    wheels stop at 3.10. An episode ends on death, level completion or the game's time
    limit, which procgen doesn't tell apart, so all of them terminate
    """

    def __init__(self, cfg: ProcgenEnvConfig) -> None:
        self.cfg = cfg

        # Note: uv provides the worker's Python 3.10 with procgen, so neverwake's own
        # environment doesn't need it
        self.cmd = [
            "uv",
            "run",
            "--no-project",
            "--python",
            "3.10",
            "--with",
            "procgen==0.10.7",
            "--with",
            "numpy<2",
            "python",
            str(Path(__file__).with_name("procgen_worker.py")),
            cfg.game,
            cfg.distribution_mode,
            str(cfg.num_levels),
            str(cfg.start_level),
        ]

        # State
        self.worker: subprocess.Popen[bytes] | None = None
        self.seed = 0
        self.restarts = 0  # Worker restarts since the last seeded reset
        self.mid_episode = False  # Whether a step happened since the episode started
        self.screen = np.zeros((64, 64, 3), dtype=np.uint8)  # Last frame

    @property
    def all_actions(self) -> torch.Tensor:
        # Note: every procgen game has the same 15 button combinations
        return torch.arange(15)

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

    def start(self, seed: int) -> None:
        self.close()
        self.worker = subprocess.Popen(
            [*self.cmd, str(seed)], stdin=subprocess.PIPE, stdout=subprocess.PIPE
        )

        # Note: skip anything the imports print before the worker is ready
        assert self.worker.stdout is not None
        while (line := self.worker.stdout.readline()) != b"ready\n":
            if not line:
                raise RuntimeError("procgen worker exited before it was ready")

    def send(self, command: bytes) -> None:
        assert self.worker is not None and self.worker.stdin is not None
        self.worker.stdin.write(command)
        self.worker.stdin.flush()

    def receive(self, num_bytes: int) -> bytes:
        assert self.worker is not None and self.worker.stdout is not None
        data = self.worker.stdout.read(num_bytes)
        if len(data) != num_bytes:
            raise RuntimeError("procgen worker stopped")
        return data

    def receive_frame(self) -> np.ndarray:
        return np.frombuffer(self.receive(64 * 64 * 3), dtype=np.uint8).reshape(
            64, 64, 3
        )

    def step(
        self, action: int
    ) -> tuple[torch.Tensor, float, bool, bool, dict[str, Any]]:
        self.send(b"s" + bytes([action]))
        done, complete, reward = struct.unpack("<??f", self.receive(6))
        frame = self.receive_frame()

        # Note: a finished episode's frame is already the next one's first, so the
        # last frame repeats like a finished ViZDoom episode
        if not done:
            self.screen = frame.copy()
        self.mid_episode = not done

        info: dict[str, Any] = {"game_end": done, "level_complete": complete}
        return self.proc_obs(self.screen), float(reward), done, False, info

    def reset(self, seed: int | None) -> tuple[torch.Tensor, dict[str, Any]]:
        # Note: procgen can't reset mid-episode, so the worker restarts on a new seed
        if seed is not None:
            self.seed, self.restarts = seed, 0
            self.start(seed)
        elif self.worker is None or self.mid_episode:
            self.restarts += 1
            self.start(self.seed + self.restarts)

        self.send(b"r")
        self.screen = self.receive_frame().copy()
        self.mid_episode = False
        return self.proc_obs(self.screen), {}

    def render(self) -> np.ndarray:
        return self.screen.copy()

    def close(self) -> None:
        if self.worker is None:
            return
        try:
            self.send(b"q")
            self.worker.wait(timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            self.worker.kill()
        self.worker = None
