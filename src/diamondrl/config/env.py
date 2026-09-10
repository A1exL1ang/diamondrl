from dataclasses import dataclass


@dataclass
class EnvConfig:
    """
    Base class for all env configs
    """

    pass


@dataclass
class AtariEnvConfig(EnvConfig):
    game: str
    max_noop: int = 30
    frameskip: int = 4
    shape: tuple[int, int, int] = (3, 64, 64)
