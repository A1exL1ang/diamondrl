from dataclasses import dataclass


@dataclass
class EnvConfig:
    """
    Base class for all env configs
    """

    pass


@dataclass
class AtariEnvConfig(EnvConfig):
    rescaled_shape: tuple[int, int, int]
    episodic_life: bool
    game: str
    max_noop: int
    frameskip: int


@dataclass
class MarioForwardRewL1EnvConfig(EnvConfig):
    rescaled_shape: tuple[int, int, int]
    max_noop: int
    frameskip: int
    block_size: int  # Pixels of rightward progress per +1 reward


@dataclass
class DoomHpSupBinRewEnvConfig(EnvConfig):
    rescaled_shape: tuple[int, int, int]
    frameskip: int


@dataclass
class DoomDeathmatchKillRewEnvConfig(EnvConfig):
    rescaled_shape: tuple[int, int, int]
    frameskip: int


@dataclass
class CraftaxClassicAchRewEnvConfig(EnvConfig):
    rescaled_shape: tuple[int, int, int]
