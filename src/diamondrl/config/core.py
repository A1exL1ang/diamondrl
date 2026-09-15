from dataclasses import dataclass

from diamondrl.config.env import EnvConfig


@dataclass
class DiffusionConfig:
    obs_shape: tuple[int, int, int]

    inp_ch: int
    inp_proj_ch: int
    midcoder_ch: int
    enc_dec_channels: tuple[int, ...]
    out_ch: int

    group_size: int
    cond_dim: int

    rho: float

    sigma_data: float
    sigma_off: float

    sigma_max: float
    sigma_min: float

    denoising_steps: int


@dataclass
class RLConfig:
    pass


@dataclass
class TrainConfig:
    pass


class Config:
    seed: int
    env: EnvConfig
    diffusion: DiffusionConfig
    rl: RLConfig
    train: TrainConfig
