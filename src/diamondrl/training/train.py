import random
from dataclasses import dataclass

import numpy as np
import torch

from diamondrl.config.core import Config
from diamondrl.data.replay_buffer import ReplayBuffer
from diamondrl.edm.edm_loss import edm_loss
from diamondrl.env.base import AbstractEnv
from diamondrl.rl.rl_loss import rl_loss
from diamondrl.rl.rollout import rollout
from diamondrl.training.build import Agent, Optimizers, build_agent, build_optimizers
from diamondrl.training.collector import Collector
from diamondrl.world.reward_end_loss import reward_end_loss


@dataclass
class TrainState:
    agent: Agent
    optimizers: Optimizers
    replay_buffer: ReplayBuffer
    collector: Collector
    epoch: int


def derive_seed(base_seed: int, epoch: int) -> int:
    return base_seed + epoch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def init_state(cfg: Config, env: AbstractEnv) -> TrainState:
    # Seed
    seed_everything(cfg.seed)

    # Agent
    agent = build_agent(cfg, env)

    # Handle device here
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent.denoiser.to(device)
    agent.reward_end_model.to(device)
    agent.actor_critic_model.to(device)

    # Other stuff
    optimizers = build_optimizers(cfg, agent)
    replay_buffer = ReplayBuffer(cfg.train.sample_weights)
    collector = Collector(
        env,
        replay_buffer,
        agent.actor_critic_model,
        cfg.train.collection.eps,
        cfg.seed,
    )
    epoch = 0

    return TrainState(
        agent=agent,
        optimizers=optimizers,
        replay_buffer=replay_buffer,
        collector=collector,
        epoch=epoch,
    )


def initial_collect(cfg: Config, state: TrainState) -> None:
    collection_cfg = cfg.train.collection
    state.collector.collect(collection_cfg.init_min_env_steps)

    # Keep collecting until we either reach our threshold or
    while (
        state.collector.reward_pos + state.collector.reward_neg
        < collection_cfg.init_min_reward_steps
        and state.collector.total_steps + collection_cfg.env_steps_per_epoch
        <= collection_cfg.init_max_env_steps
    ):
        state.collector.collect(collection_cfg.env_steps_per_epoch)


def train(cfg: Config, state: TrainState) -> None:
    device = next(state.agent.denoiser.parameters()).device

    while True:
        is_first = state.epoch == 0

        # Collection
        if is_first:
            initial_collect(cfg, state)
        else:
            if (
                state.collector.total_steps + cfg.train.collection.env_steps_per_epoch
                > cfg.train.collection.total_env_steps
            ):
                break
            state.collector.collect(cfg.train.collection.env_steps_per_epoch)

        # Diffusion model
        diffusion_cfg = cfg.train.diffusion
        for step in range(
            diffusion_cfg.steps_init if is_first else diffusion_cfg.steps_per_epoch
        ):
            context_batch = state.replay_buffer.sample_context(
                batch_size=diffusion_cfg.batch_size,
                ctx_frames=cfg.world.ctx_frames,
                future_frames=diffusion_cfg.future_frames,
            ).to(device)
            loss_diffusion, loss_list_diffusion = edm_loss(
                denoiser=state.agent.denoiser,
                context_batch=context_batch,
                sigma_min_train=diffusion_cfg.sigma_min,
                sigma_max_train=diffusion_cfg.sigma_max,
                p_mean=diffusion_cfg.p_mean,
                p_std=diffusion_cfg.p_std,
            )
            state.optimizers.denoiser.update(loss_diffusion)

        # Reward model
        reward_end_cfg = cfg.train.reward_end
        for step in range(
            reward_end_cfg.steps_init if is_first else reward_end_cfg.steps_per_epoch
        ):
            seq_batch = state.replay_buffer.sample_masked_sequence(
                batch_size=reward_end_cfg.batch_size,
                seq_transition_count=cfg.world.ctx_frames + cfg.world.horizon,
            ).to(device)
            loss_reward_end, loss_reward_end_info = reward_end_loss(
                state.agent.reward_end_model, seq_batch
            )
            state.optimizers.reward_end.update(loss_reward_end)

        # RL
        # TODO: consistent naming of loss_<name>
        actor_critic_cfg = cfg.train.actor_critic
        for step in range(
            actor_critic_cfg.steps_init
            if is_first
            else actor_critic_cfg.steps_per_epoch
        ):
            context_batch = state.replay_buffer.sample_context(
                batch_size=actor_critic_cfg.batch_size,
                ctx_frames=cfg.world.ctx_frames,
                future_frames=0,
            ).to(device)

            trajectories = rollout(
                actor_critic_model=state.agent.actor_critic_model,
                reward_end_model=state.agent.reward_end_model,
                next_obs_model=state.agent.next_obs_model,
                obs_ctx_stk_init=context_batch.obs_stack,
                action_seq_init=context_batch.actions,
                num_rollout_actions=cfg.world.horizon,
            ).to(device)

            loss_rl, loss_rl_info = rl_loss(
                traj=trajectories,
                weight_value=cfg.train.rl.weight_value,
                weight_entropy=cfg.train.rl.weight_entropy,
                lamb=cfg.train.rl.lamb,
                gamma=cfg.train.rl.gamma,
            )
            state.optimizers.actor_critic.update(loss_rl)

        # Telemetry
        state.epoch += 1

    # TODO: final pass
