import random
from dataclasses import dataclass

import numpy as np
import torch

from neverwake.config.core import Config
from neverwake.data.replay_buffer import ReplayBuffer
from neverwake.env.base import AbstractEnv
from neverwake.rl.actor_critic_loss import actor_critic_loss
from neverwake.rl.rollout import rollout
from neverwake.training.build import Agent, build_agent, build_optimizers
from neverwake.training.checkpoint import Checkpointer
from neverwake.training.collector import Collector
from neverwake.training.eval import Evaluator
from neverwake.training.optim import Optimizers
from neverwake.training.telemetry import Telemetry
from neverwake.world.denoiser_loss import denoiser_loss
from neverwake.world.reward_end_loss import reward_end_loss


@dataclass
class TrainState:
    agent: Agent
    optimizers: Optimizers
    replay_buffer: ReplayBuffer
    collector: Collector
    evaluator: Evaluator
    epoch: int
    epochs_after_collection: int


def derive_seed(base_seed: int, epoch: int, is_eval: bool) -> int:
    return int(np.random.SeedSequence([base_seed, epoch, is_eval]).generate_state(1)[0])


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def init_state(
    cfg: Config, train_env: AbstractEnv, eval_env: AbstractEnv, epoch: int
) -> TrainState:
    # Seed (a resumed run reseeds from its epoch so it doesn't replay the start)
    derived_seed_train = derive_seed(base_seed=cfg.seed, epoch=epoch, is_eval=False)
    derived_seed_eval = derive_seed(base_seed=cfg.seed, epoch=epoch, is_eval=True)
    seed_everything(derived_seed_train)

    # Agent
    agent = build_agent(cfg, train_env)

    # Handle device here
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent.denoiser.to(device)
    agent.reward_end.to(device)
    agent.actor_critic.to(device)

    # Compile in-place
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        agent.denoiser.compile(mode="reduce-overhead", dynamic=False)
        agent.reward_end.compile(mode="reduce-overhead", dynamic=False)
        agent.actor_critic.compile(mode="reduce-overhead", dynamic=False)

    # Other stuff
    optimizers = build_optimizers(cfg, agent)
    replay_buffer = ReplayBuffer(cfg.train.sample_weights)

    # Seed for the env is set here
    collector = Collector(
        env=train_env,
        replay_buffer=replay_buffer,
        actor_critic=agent.actor_critic,
        eps=cfg.train.collection.eps,
        seed=derived_seed_train,
    )

    # Seed for the evaluator env is also sethere
    evaluator = Evaluator(
        env=eval_env,
        actor_critic=agent.actor_critic,
        eps=cfg.train.eval.eps,
        max_steps_per_game=cfg.train.eval.max_steps_per_game,
        seed=derived_seed_eval,
    )

    return TrainState(
        agent=agent,
        optimizers=optimizers,
        replay_buffer=replay_buffer,
        collector=collector,
        evaluator=evaluator,
        epoch=epoch,
        epochs_after_collection=0,
    )


def initial_collect(cfg: Config, state: TrainState) -> None:
    # Note that data is stamped with the epoch it is logged under, which is one more
    # than state.epoch since that counts completed epochs
    collection_cfg = cfg.train.collection
    state.collector.collect(collection_cfg.init_min_env_steps, epoch=state.epoch + 1)

    # Keep collecting until we either reach our threshold or
    while (
        state.collector.reward_pos + state.collector.reward_neg
        < collection_cfg.init_min_reward_steps
        and state.collector.env_steps + collection_cfg.env_steps_per_epoch
        <= collection_cfg.init_max_env_steps
    ):
        state.collector.collect(
            collection_cfg.env_steps_per_epoch, epoch=state.epoch + 1
        )


def train(
    cfg: Config, state: TrainState, tel: Telemetry, checkpointer: Checkpointer
) -> None:
    device = next(state.agent.denoiser.parameters()).device
    use_bf16 = device.type == "cuda"

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
                cfg.train.collection.env_steps_per_epoch, epoch=state.epoch + 1
            )
        elif state.epochs_after_collection < cfg.train.epochs_after_collection:
            state.epochs_after_collection += 1
        else:
            break

        tel.lap("collect")

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

            torch.compiler.cudagraph_mark_step_begin()

            with torch.autocast(device.type, torch.bfloat16, enabled=use_bf16):
                loss_denoiser, info_denoiser = denoiser_loss(
                    denoiser=state.agent.denoiser,
                    context_batch=context_batch,
                    sigma_min=denoiser_cfg.sigma_min,
                    sigma_max=denoiser_cfg.sigma_max,
                    p_mean=denoiser_cfg.p_mean,
                    p_std=denoiser_cfg.p_std,
                )
            grad_norm = state.optimizers.denoiser.update(loss_denoiser)
            tel.update("denoiser", loss_denoiser, info_denoiser, grad_norm)

        tel.lap("denoiser")

        # Reward end
        reward_end_cfg = cfg.train.reward_end
        for _ in range(
            reward_end_cfg.steps_init if is_first else reward_end_cfg.steps_per_epoch
        ):
            seq_batch = state.replay_buffer.sample_masked_sequence(
                batch_size=reward_end_cfg.batch_size,
                num_transitions=cfg.world.ctx_frames + cfg.world.horizon,
            ).to(device)

            torch.compiler.cudagraph_mark_step_begin()

            with torch.autocast(device.type, torch.bfloat16, enabled=use_bf16):
                loss_reward_end, info_reward_end = reward_end_loss(
                    state.agent.reward_end, seq_batch
                )

            grad_norm = state.optimizers.reward_end.update(loss_reward_end)
            tel.update("reward_end", loss_reward_end, info_reward_end, grad_norm)

        tel.lap("reward_end")

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

            torch.compiler.cudagraph_mark_step_begin()

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
            tel.imagination(traj)

            grad_norm = state.optimizers.actor_critic.update(loss_actor_critic)

            tel.update("actor_critic", loss_actor_critic, info_actor_critic, grad_norm)

        tel.lap("actor_critic")

        # Telemetry and eval
        state.epoch += 1
        tel.end_epoch(state.epoch)
        checkpointer.save(state.epoch, state.epochs_after_collection)

    # Final eval
    tel.final_eval()
    checkpointer.save_final()
