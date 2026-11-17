from dataclasses import dataclass


@dataclass
class CollectionConfig:
    eps: float
    temperature: float
    init_min_env_steps: int
    init_max_env_steps: int
    init_min_reward_steps: int
    env_steps_per_epoch: int
    total_env_steps: int


@dataclass
class WorldModelTrainConfig:
    steps_init: int
    steps_per_epoch: int
    batch_size: int
    lr: float
    weight_decay: float
    warmup_steps: int
    max_grad_norm: float

    # Denoiser
    future_frames: int
    p_mean: float
    p_std: float
    sigma_min: float
    sigma_max: float


@dataclass
class ActorCriticTrainConfig:
    steps_init: int
    steps_per_epoch: int
    batch_size: int
    lr: float
    weight_decay: float
    warmup_steps: int
    max_grad_norm: float


@dataclass
class RLConfig:
    gamma: float
    lamb: float
    weight_value: float
    weight_entropy: float

    # The running range of the returns, which the critic and the advantages are in
    # units of
    return_percentiles: tuple[float, float]  # Its low and high end
    return_range_decay: float  # How slowly it follows new batches
    return_range_min: float  # A smaller range counts as this


@dataclass
class EvalConfig:
    every: int
    num_games: int
    num_games_final: int
    max_steps_per_game: int
    eps: float


@dataclass
class TrainConfig:
    sample_weights: tuple[float, ...]
    epochs_after_collection: int

    collection: CollectionConfig
    world_model: WorldModelTrainConfig
    actor_critic: ActorCriticTrainConfig
    rl: RLConfig
    eval: EvalConfig
