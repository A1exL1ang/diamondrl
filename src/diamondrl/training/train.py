import random
from dataclasses import dataclass

import numpy as np
import torch

from diamondrl.config.core import Config
from diamondrl.data.replay_buffer import ReplayBuffer
from diamondrl.env.base import AbstractEnv
from diamondrl.rl.actor_critic_loss import actor_critic_loss
from diamondrl.rl.rollout import rollout
from diamondrl.training.build import Agent, build_agent, build_optimizers
from diamondrl.training.collector import Collector
from diamondrl.training.optim import Optimizers
from diamondrl.world.denoiser_loss import denoiser_loss
from diamondrl.world.reward_end_loss import reward_end_loss


@dataclass
class TrainState:
    agent: Agent
    optimizers: Optimizers
    replay_buffer: ReplayBuffer
    collector: Collector
    epoch: int
    epochs_after_collection: int


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
    agent.reward_end.to(device)
    agent.actor_critic.to(device)

    # Other stuff
    optimizers = build_optimizers(cfg, agent)
    replay_buffer = ReplayBuffer(cfg.train.sample_weights)
    collector = Collector(
        env,
        replay_buffer,
        agent.actor_critic,
        cfg.train.collection.eps,
        cfg.seed,
    )

    return TrainState(
        agent=agent,
        optimizers=optimizers,
        replay_buffer=replay_buffer,
        collector=collector,
        epoch=0,
        epochs_after_collection=0,
    )


def initial_collect(cfg: Config, state: TrainState) -> None:
    collection_cfg = cfg.train.collection
    state.collector.collect(collection_cfg.init_min_env_steps, epoch=state.epoch)

    # Keep collecting until we either reach our threshold or
    while (
        state.collector.reward_pos + state.collector.reward_neg
        < collection_cfg.init_min_reward_steps
        and state.collector.env_steps + collection_cfg.env_steps_per_epoch
        <= collection_cfg.init_max_env_steps
    ):
        state.collector.collect(collection_cfg.env_steps_per_epoch, epoch=state.epoch)


def train(cfg: Config, state: TrainState) -> None:
    device = next(state.agent.denoiser.parameters()).device

    while True:
        is_first = state.epoch == 0

        # Collection (warning: we increment epochs_after_collection here upfront)
        if is_first:
            initial_collect(cfg, state)
        elif (
            state.collector.env_steps + cfg.train.collection.env_steps_per_epoch
            <= cfg.train.collection.total_env_steps
        ):
            state.collector.collect(
                cfg.train.collection.env_steps_per_epoch, epoch=state.epoch
            )
        elif state.epochs_after_collection < cfg.train.epochs_after_collection:
            state.epochs_after_collection += 1
        else:
            break

        # Denoiser
        denoiser_cfg = cfg.train.denoiser
        for _ in range(
            denoiser_cfg.steps_init if is_first else denoiser_cfg.steps_per_epoch
        ):
            context_batch = state.replay_buffer.sample_context(
                batch_size=denoiser_cfg.batch_size,
                ctx_frames=cfg.world.ctx_frames,
                future_frames=denoiser_cfg.future_frames,
            ).to(device)
            loss_denoiser, info_denoiser = denoiser_loss(
                denoiser=state.agent.denoiser,
                context_batch=context_batch,
                sigma_min=denoiser_cfg.sigma_min,
                sigma_max=denoiser_cfg.sigma_max,
                p_mean=denoiser_cfg.p_mean,
                p_std=denoiser_cfg.p_std,
            )
            state.optimizers.denoiser.update(loss_denoiser)

        # Reward end
        reward_end_cfg = cfg.train.reward_end
        for _ in range(
            reward_end_cfg.steps_init if is_first else reward_end_cfg.steps_per_epoch
        ):
            seq_batch = state.replay_buffer.sample_masked_sequence(
                batch_size=reward_end_cfg.batch_size,
                num_transitions=cfg.world.ctx_frames + cfg.world.horizon,
            ).to(device)
            loss_reward_end, info_reward_end = reward_end_loss(
                state.agent.reward_end, seq_batch
            )
            state.optimizers.reward_end.update(loss_reward_end)

        # Actor critic
        actor_critic_cfg = cfg.train.actor_critic
        for _ in range(
            actor_critic_cfg.steps_init
            if is_first
            else actor_critic_cfg.steps_per_epoch
        ):
            context_batch = state.replay_buffer.sample_context(
                batch_size=actor_critic_cfg.batch_size,
                ctx_frames=cfg.world.ctx_frames,
                future_frames=0,
            ).to(device)

            traj = rollout(
                actor_critic=state.agent.actor_critic,
                reward_end=state.agent.reward_end,
                next_obs_model=state.agent.next_obs_model,
                obs_ctx_stack_init=context_batch.obs_stack,
                action_seq_init=context_batch.actions,
                horizon=cfg.world.horizon,
            ).to(device)

            loss_actor_critic, info_actor_critic = actor_critic_loss(
                traj=traj,
                weight_value=cfg.train.rl.weight_value,
                weight_entropy=cfg.train.rl.weight_entropy,
                lamb=cfg.train.rl.lamb,
                gamma=cfg.train.rl.gamma,
            )
            state.optimizers.actor_critic.update(loss_actor_critic)

        # Telemetry
        state.epoch += 1
