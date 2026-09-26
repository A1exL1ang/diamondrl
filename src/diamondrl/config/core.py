from dataclasses import dataclass

from diamondrl.config.env import EnvConfig


@dataclass
class DiffusionConfig:
    inp_proj_ch: int
    midcoder_ch: int
    enc_dec_channels: tuple[int, ...]
    out_ch: int

    group_size: int
    cond_dim: int

    rho: float

    sigma_data: float
    sigma_off: float

    sigma_min_train: float
    sigma_max_train: float

    sigma_max_ode: float
    sigma_min_ode: float

    denoising_steps: int


@dataclass
class RewardEndConfig:
    inp_proj_ch: int
    channels: tuple[int, ...]
    downsample_after: tuple[bool, ...]
    lstm_hidden: int
    group_size: int
    cond_dim: int


@dataclass
class ActorCriticConfig:
    inp_proj_ch: int
    channels: tuple[int, ...]
    downsample_after: tuple[bool, ...]
    lstm_hidden: int
    group_size: int


@dataclass
class WorldConfig:
    ctx_frames: int
    horizon: int


@dataclass
class TrainConfig:
    # Buffer
    sample_weights: tuple[float, ...]

    # RL
    num_edm_train_rollout: int
    weight_value: float
    weight_entropy: float
    lamb: float
    gamma: float


@dataclass
class Config:
    seed: int

    env: EnvConfig
    diffusion: DiffusionConfig

    rew_end: RewardEndConfig
    act_crit: ActorCriticConfig

    world: WorldConfig

    train: TrainConfig
