from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR

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


class ModelOptimizer:
    def __init__(
        self,
        model: nn.Module,
        lr: float,
        weight_decay: float,
        warmup_steps: int,
        max_grad_norm: float,
    ):
        self.model = model

        self.opt = AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        self.schedule = LinearLR(
            self.opt,
            start_factor=1 / warmup_steps,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        self.max_grad_norm = max_grad_norm
        self.num_updates = 0

    def update(self, loss: torch.Tensor) -> torch.Tensor:
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(
            self.model.parameters(), self.max_grad_norm
        )
        self.opt.step()
        self.schedule.step()
        self.num_updates += 1
        return grad_norm

    def state_dict(self) -> dict[str, Any]:
        return {
            "opt": self.opt.state_dict(),
            "schedule": self.schedule.state_dict(),
            "num_updates": self.num_updates,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.opt.load_state_dict(state_dict["opt"])
        self.schedule.load_state_dict(state_dict["schedule"])
        self.num_updates = state_dict["num_updates"]


@dataclass
class Optimizers:
    denoiser: ModelOptimizer
    reward_end: ModelOptimizer
    actor_critic: ModelOptimizer


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
        inp_proj_ch=cfg.reward_end.inp_proj_ch,
        channels=cfg.reward_end.channels,
        downsample_after=cfg.reward_end.downsample_after,
        lstm_hidden=cfg.reward_end.lstm_hidden,
        group_size=cfg.reward_end.group_size,
        num_actions=num_actions,
        cond_dim=cfg.reward_end.cond_dim,
    )

    actor_critic_model = ActorCriticModel(
        inp_shape=(frame_ch, frame_h, frame_w),
        inp_proj_ch=cfg.actor_critic.inp_proj_ch,
        channels=cfg.actor_critic.channels,
        downsample_after=cfg.actor_critic.downsample_after,
        lstm_hidden=cfg.actor_critic.lstm_hidden,
        num_actions=num_actions,
        group_size=cfg.actor_critic.group_size,
    )

    return Agent(
        denoiser=denoiser,
        next_obs_model=next_obs_model,
        reward_end_model=reward_end_model,
        actor_critic_model=actor_critic_model,
    )


def build_optimizers(cfg: Config, agent: Agent) -> Optimizers:
    denoiser_optimizer = ModelOptimizer(
        model=agent.denoiser,
        lr=cfg.train.diffusion.lr,
        weight_decay=cfg.train.diffusion.weight_decay,
        warmup_steps=cfg.train.diffusion.warmup_steps,
        max_grad_norm=cfg.train.diffusion.max_grad_norm,
    )
    reward_end_optimizer = ModelOptimizer(
        model=agent.reward_end_model,
        lr=cfg.train.reward_end.lr,
        weight_decay=cfg.train.reward_end.weight_decay,
        warmup_steps=cfg.train.reward_end.warmup_steps,
        max_grad_norm=cfg.train.reward_end.max_grad_norm,
    )
    actor_critic_optimizer = ModelOptimizer(
        model=agent.actor_critic_model,
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
