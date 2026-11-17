from dataclasses import dataclass

from neverwake.config.core import Config
from neverwake.env.base import AbstractEnv
from neverwake.modules.embedding import CondEmbedder
from neverwake.modules.unet import Unet
from neverwake.rl.actor_critic import ActorCriticModel
from neverwake.training.optim import ModelOptimizer, Optimizers
from neverwake.world.denoiser import Denoiser
from neverwake.world.memory import MemoryModel
from neverwake.world.next_obs_model import NextObsModel
from neverwake.world.reward_end_model import RewardEndModel
from neverwake.world.world_model import WorldModel


@dataclass
class Agent:
    world_model: WorldModel
    next_obs_model: NextObsModel
    actor_critic: ActorCriticModel


def build_agent(cfg: Config, env: AbstractEnv) -> Agent:
    frame_ch, frame_h, frame_w = env.obs_shape
    num_actions = env.all_actions.shape[0]

    # TODO: have some value errors here to enforce some basic constraints
    # Or enforce the group size constraints upon construction actually

    memory = MemoryModel(
        inp_shape=(frame_ch, frame_h, frame_w),
        inp_proj_ch=cfg.memory.inp_proj_ch,
        channels=cfg.memory.channels,
        downsample_after=cfg.memory.downsample_after,
        group_size=cfg.memory.group_size,
        num_actions=num_actions,
        action_dim=cfg.memory.action_dim,
        mamba_dim=cfg.memory.mamba_dim,
        mamba_layers=cfg.memory.mamba_layers,
    )

    u_net = Unet(
        inp_ch=frame_ch * (cfg.world.ctx_frames + 1),
        inp_proj_ch=cfg.denoiser.inp_proj_ch,
        midcoder_ch=cfg.denoiser.midcoder_ch,
        enc_dec_channels=cfg.denoiser.enc_dec_channels,
        out_ch=frame_ch,
        group_size=cfg.denoiser.group_size,
        cond_dim=cfg.denoiser.cond_dim,
        num_heads=cfg.denoiser.num_heads,
    )

    cond_embedder = CondEmbedder(
        cond_dim=cfg.denoiser.cond_dim,
        memory_dim=cfg.memory.mamba_dim,
        w_min=cfg.denoiser.fourier_w_min,
        w_max=cfg.denoiser.fourier_w_max,
    )

    denoiser = Denoiser(
        net=u_net,
        cond_embedder=cond_embedder,
        sigma_off=cfg.denoiser.sigma_off,
        sigma_data=cfg.denoiser.sigma_data,
    )

    reward_end = RewardEndModel(
        memory_dim=cfg.memory.mamba_dim,
        num_bins=cfg.world.reward_bins,
        bin_limit=cfg.world.reward_bin_limit,
    )

    world_model = WorldModel(memory=memory, denoiser=denoiser, reward_end=reward_end)

    next_obs_model = NextObsModel(
        denoiser=denoiser,
        obs_shape=(frame_ch, frame_h, frame_w),
        rho=cfg.denoiser.rho,
        sigma_min_ode=cfg.denoiser.sigma_min_ode,
        sigma_max_ode=cfg.denoiser.sigma_max_ode,
        denoising_steps=cfg.denoiser.denoising_steps,
    )

    actor_critic = ActorCriticModel(
        inp_shape=(frame_ch, frame_h, frame_w),
        inp_proj_ch=cfg.actor_critic.inp_proj_ch,
        channels=cfg.actor_critic.channels,
        downsample_after=cfg.actor_critic.downsample_after,
        lstm_dim=cfg.actor_critic.lstm_dim,
        num_actions=num_actions,
        group_size=cfg.actor_critic.group_size,
        return_percentiles=cfg.train.rl.return_percentiles,
        return_range_decay=cfg.train.rl.return_range_decay,
        return_range_min=cfg.train.rl.return_range_min,
    )

    return Agent(
        world_model=world_model,
        next_obs_model=next_obs_model,
        actor_critic=actor_critic,
    )


def build_optimizers(cfg: Config, agent: Agent) -> Optimizers:
    world_model_optimizer = ModelOptimizer(
        model=agent.world_model,
        lr=cfg.train.world_model.lr,
        weight_decay=cfg.train.world_model.weight_decay,
        warmup_steps=cfg.train.world_model.warmup_steps,
        max_grad_norm=cfg.train.world_model.max_grad_norm,
    )
    actor_critic_optimizer = ModelOptimizer(
        model=agent.actor_critic,
        lr=cfg.train.actor_critic.lr,
        weight_decay=cfg.train.actor_critic.weight_decay,
        warmup_steps=cfg.train.actor_critic.warmup_steps,
        max_grad_norm=cfg.train.actor_critic.max_grad_norm,
    )
    return Optimizers(
        world_model=world_model_optimizer,
        actor_critic=actor_critic_optimizer,
    )
