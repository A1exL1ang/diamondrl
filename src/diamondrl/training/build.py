from dataclasses import dataclass

from diamondrl.config.core import Config
from diamondrl.env.base import AbstractEnv
from diamondrl.modules.embedding import CondEmbedder
from diamondrl.modules.unet import Unet
from diamondrl.rl.actor_critic import ActorCriticModel
from diamondrl.training.optim import ModelOptimizer, Optimizers
from diamondrl.world.denoiser import Denoiser
from diamondrl.world.next_obs_model import NextObsModel
from diamondrl.world.reward_end_model import RewardEndModel


@dataclass
class Agent:
    denoiser: Denoiser
    next_obs_model: NextObsModel
    reward_end: RewardEndModel
    actor_critic: ActorCriticModel


def build_agent(cfg: Config, env: AbstractEnv) -> Agent:
    frame_ch, frame_h, frame_w = env.obs_shape
    num_actions = env.all_actions.shape[0]

    # TODO: have some value errors here to enforce some basic constraints
    # Or enforce the group size constraints upon construction actually

    u_net = Unet(
        inp_ch=frame_ch * (cfg.world.ctx_frames + 1),
        inp_proj_ch=cfg.denoiser.inp_proj_ch,
        midcoder_ch=cfg.denoiser.midcoder_ch,
        enc_dec_channels=cfg.denoiser.enc_dec_channels,
        out_ch=frame_ch,
        group_size=cfg.denoiser.group_size,
        cond_dim=cfg.denoiser.cond_dim,
    )

    cond_embedder = CondEmbedder(
        cond_dim=cfg.denoiser.cond_dim,
        ctx_steps=cfg.world.ctx_frames,
        num_actions=num_actions,
    )

    denoiser = Denoiser(
        net=u_net,
        cond_embedder=cond_embedder,
        sigma_off=cfg.denoiser.sigma_off,
        sigma_data=cfg.denoiser.sigma_data,
    )

    next_obs_model = NextObsModel(
        denoiser=denoiser,
        obs_shape=(frame_ch, frame_h, frame_w),
        rho=cfg.denoiser.rho,
        sigma_min_ode=cfg.denoiser.sigma_min_ode,
        sigma_max_ode=cfg.denoiser.sigma_max_ode,
        denoising_steps=cfg.denoiser.denoising_steps,
    )

    reward_end = RewardEndModel(
        inp_shape=(frame_ch * 2, frame_h, frame_w),
        inp_proj_ch=cfg.reward_end.inp_proj_ch,
        channels=cfg.reward_end.channels,
        downsample_after=cfg.reward_end.downsample_after,
        lstm_dim=cfg.reward_end.lstm_dim,
        group_size=cfg.reward_end.group_size,
        num_actions=num_actions,
        cond_dim=cfg.reward_end.cond_dim,
    )

    actor_critic = ActorCriticModel(
        inp_shape=(frame_ch, frame_h, frame_w),
        inp_proj_ch=cfg.actor_critic.inp_proj_ch,
        channels=cfg.actor_critic.channels,
        downsample_after=cfg.actor_critic.downsample_after,
        lstm_dim=cfg.actor_critic.lstm_dim,
        num_actions=num_actions,
        group_size=cfg.actor_critic.group_size,
    )

    return Agent(
        denoiser=denoiser,
        next_obs_model=next_obs_model,
        reward_end=reward_end,
        actor_critic=actor_critic,
    )


def build_optimizers(cfg: Config, agent: Agent) -> Optimizers:
    denoiser_optimizer = ModelOptimizer(
        model=agent.denoiser,
        lr=cfg.train.denoiser.lr,
        weight_decay=cfg.train.denoiser.weight_decay,
        warmup_steps=cfg.train.denoiser.warmup_steps,
        max_grad_norm=cfg.train.denoiser.max_grad_norm,
    )
    reward_end_optimizer = ModelOptimizer(
        model=agent.reward_end,
        lr=cfg.train.reward_end.lr,
        weight_decay=cfg.train.reward_end.weight_decay,
        warmup_steps=cfg.train.reward_end.warmup_steps,
        max_grad_norm=cfg.train.reward_end.max_grad_norm,
    )
    actor_critic_optimizer = ModelOptimizer(
        model=agent.actor_critic,
        lr=cfg.train.actor_critic.lr,
        weight_decay=cfg.train.actor_critic.weight_decay,
        warmup_steps=cfg.train.actor_critic.warmup_steps,
        max_grad_norm=cfg.train.actor_critic.max_grad_norm,
    )
    return Optimizers(
        denoiser=denoiser_optimizer,
        reward_end=reward_end_optimizer,
        actor_critic=actor_critic_optimizer,
    )
