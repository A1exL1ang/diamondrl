from dataclasses import dataclass

from diamondrl.config.core import Config
from diamondrl.edm.denoiser import Denoiser
from diamondrl.env.base import AbstractEnv
from diamondrl.models.actor_critic_model import ActorCriticModel
from diamondrl.models.embedding import CondEmbedder
from diamondrl.models.reward_end_model import RewardEndModel
from diamondrl.models.unet import Unet
from diamondrl.world.next_obs_model import NextObsModel


@dataclass
class Agent:
    denoiser: Denoiser
    next_obs_model: NextObsModel
    reward_end_model: RewardEndModel
    actor_critic_model: ActorCriticModel


def build_agent(cfg: Config, env: AbstractEnv) -> Agent:
    frame_ch, frame_h, frame_w = env.obs_shape
    num_actions = env.all_actions.shape[0]

    # TODO: have some value errors here to enforce some basic constraints
    # Or enforce the group size constraints upon construction actually

    u_net = Unet(
        inp_ch=frame_ch * (cfg.world.ctx_frames + 1),
        inp_proj_ch=cfg.diffusion.inp_proj_ch,
        midcoder_ch=cfg.diffusion.midcoder_ch,
        enc_dec_channels=cfg.diffusion.enc_dec_channels,
        out_ch=frame_ch,
        group_size=cfg.diffusion.group_size,
        cond_dim=cfg.diffusion.cond_dim,
    )

    cond_embedder = CondEmbedder(
        cond_dim=cfg.diffusion.cond_dim,
        ctx_steps=cfg.world.ctx_frames,
        num_actions=num_actions,
    )

    denoiser = Denoiser(
        edm_f_net=u_net,
        cond_embedder=cond_embedder,
        sigma_off=cfg.diffusion.sigma_off,
        sigma_data=cfg.diffusion.sigma_data,
    )

    next_obs_model = NextObsModel(
        denoiser=denoiser,
        obs_shape=(frame_ch, frame_h, frame_w),
        rho=cfg.diffusion.rho,
        sigma_max_ode=cfg.diffusion.sigma_max_ode,
        sigma_min_ode=cfg.diffusion.sigma_min_ode,
        steps=cfg.diffusion.denoising_steps,
    )

    reward_end_model = RewardEndModel(
        inp_shape=(frame_ch * 2, frame_h, frame_w),
        inp_proj_ch=cfg.rew_end.inp_proj_ch,
        channels=cfg.rew_end.channels,
        downsample_after=cfg.rew_end.downsample_after,
        lstm_hidden=cfg.rew_end.lstm_hidden,
        group_size=cfg.rew_end.group_size,
        num_actions=num_actions,
        cond_dim=cfg.rew_end.cond_dim,
    )

    actor_critic_model = ActorCriticModel(
        inp_shape=(frame_ch, frame_h, frame_w),
        inp_proj_ch=cfg.act_crit.inp_proj_ch,
        channels=cfg.act_crit.channels,
        downsample_after=cfg.act_crit.downsample_after,
        lstm_hidden=cfg.act_crit.lstm_hidden,
        num_actions=num_actions,
        group_size=cfg.act_crit.group_size,
    )

    return Agent(
        denoiser=denoiser,
        next_obs_model=next_obs_model,
        reward_end_model=reward_end_model,
        actor_critic_model=actor_critic_model,
    )
