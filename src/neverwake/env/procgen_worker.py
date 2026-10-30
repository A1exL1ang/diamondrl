# pyright: reportMissingImports=false
"""
Worker for neverwake's ProcgenEnv, run by a Python that has procgen (its wheels stop at
Python 3.10), so it only uses the standard library, numpy and procgen.

Arguments: game distribution_mode num_levels start_level seed. After the env is built it
writes b"ready\\n", then answers commands on stdin:
- b"r": the current episode's first frame
- b"s" + one action byte: done, level complete, reward (struct "<??f"), then the frame
- anything else: exit
Frames are raw (64, 64, 3) uint8.
"""

import struct
import sys

import numpy as np
from procgen import ProcgenEnv  # type: ignore[import-not-found]


def main() -> None:
    game, distribution_mode = sys.argv[1], sys.argv[2]
    num_levels, start_level, seed = (int(arg) for arg in sys.argv[3:6])
    env = ProcgenEnv(
        num_envs=1,
        env_name=game,
        distribution_mode=distribution_mode,
        num_levels=num_levels,
        start_level=start_level,
        rand_seed=seed,
    )
    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
    stdout.write(b"ready\n")
    stdout.flush()

    # Note: procgen resets itself when an episode ends, so the frame returned with done
    # is already the next episode's first and is kept for the next reset
    first = env.reset()["rgb"][0]
    while True:
        command = stdin.read(1)
        if command == b"r":
            stdout.write(first.tobytes())
        elif command == b"s":
            action = stdin.read(1)[0]
            obs, reward, done, info = env.step(np.array([action]))
            frame = obs["rgb"][0]
            if done[0]:
                first = frame
            complete = bool(info[0].get("prev_level_complete", 0))
            header = struct.pack("<??f", bool(done[0]), complete, float(reward[0]))
            stdout.write(header + frame.tobytes())
        else:
            return
        stdout.flush()


if __name__ == "__main__":
    main()
