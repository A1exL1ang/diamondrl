from pathlib import Path
from typing import Any

import dacite
import yaml

from neverwake.config.core import Config
from neverwake.config.env import (
    AtariEnvConfig,
    DoomHpSupBinRewEnvConfig,
    EnvConfig,
    MarioForwardRewL1EnvConfig,
)


def load_config(path: Path) -> Config:
    with open(path) as f:
        data = yaml.safe_load(f)
    return dacite.from_dict(Config, data, config=dacite_config())


def dacite_config() -> dacite.Config:
    return dacite.Config(
        cast=[tuple, float], strict=True, type_hooks={EnvConfig: env_config}
    )


def env_config(data: dict[str, Any]) -> EnvConfig:
    env_configs = {
        "atari": AtariEnvConfig,
        "mario_forward_rew_l1": MarioForwardRewL1EnvConfig,
        "doom_hp_sup_bin_rew": DoomHpSupBinRewEnvConfig,
    }
    env_fields = {k: v for k, v in data.items() if k != "type"}

    return dacite.from_dict(
        env_configs[data["type"]], env_fields, config=dacite_config()
    )
