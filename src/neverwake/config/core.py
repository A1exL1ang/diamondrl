from dataclasses import dataclass
from typing import Literal

from neverwake.config.env import EnvConfig
from neverwake.config.train import TrainConfig


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
class TelemetryConfig:
    project: str
    mode: Literal["online", "offline", "disabled"]
    flush_every: int  # updates per model between loss rows
    video_every: int  # epochs
    video_max_frames: int
    video_clips: int  # real games and imagined rollouts shown side by side
    video_scale: int  # Whole-number upscale of the frames the model sees
    save_every: int  # epochs between permanent weight snapshots
    stats_interval_sec: float  # how often W&B samples GPU/CPU utilization


@dataclass
class Config:
    seed: int

    env: EnvConfig
    denoiser: DenoiserConfig
    reward_end: RewardEndConfig
    actor_critic: ActorCriticConfig
    world: WorldConfig
    train: TrainConfig
    telemetry: TelemetryConfig
