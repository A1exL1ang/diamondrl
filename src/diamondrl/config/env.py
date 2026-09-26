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
    game: str
    max_noop: int
    frameskip: int
