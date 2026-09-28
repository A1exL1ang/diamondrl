from dataclasses import dataclass

from diamondrl.config.env import EnvConfig
from diamondrl.config.train import TrainConfig


@dataclass
class DenoiserConfig:
    inp_proj_ch: int
    midcoder_ch: int
    enc_dec_channels: tuple[int, ...]

    group_size: int
    cond_dim: int

    rho: float

    sigma_data: float
    sigma_off: float

    sigma_min_ode: float
    sigma_max_ode: float

    denoising_steps: int


@dataclass
class RewardEndConfig:
    inp_proj_ch: int
    channels: tuple[int, ...]
    downsample_after: tuple[bool, ...]
    lstm_dim: int
    group_size: int
    cond_dim: int


@dataclass
class ActorCriticConfig:
    inp_proj_ch: int
    channels: tuple[int, ...]
    downsample_after: tuple[bool, ...]
    lstm_dim: int
    group_size: int


@dataclass
class WorldConfig:
    ctx_frames: int
    horizon: int


@dataclass
class Config:
    seed: int

    env: EnvConfig
    denoiser: DenoiserConfig
    reward_end: RewardEndConfig
    actor_critic: ActorCriticConfig
    world: WorldConfig
    train: TrainConfig
