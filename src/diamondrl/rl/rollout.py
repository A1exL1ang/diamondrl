from dataclasses import dataclass

import torch
from torch.distributions import Categorical

from diamondrl.models.actor_critic_model import ActorCriticModel
from diamondrl.models.reward_end_model import RewardEndModel
from diamondrl.world.next_obs_model import NextObsModel
from diamondrl.world.world_stepper import BatchedWorldStepper


@dataclass
class Trajectory:
    """
    obs and value has an additional entry for the true final frame
    """

    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    ends: torch.Tensor
    valid: torch.Tensor

    policy_logits: torch.Tensor
    obs_values: torch.Tensor

    def to(self, device: torch.device | str) -> "Trajectory":
        return Trajectory(
            obs=self.obs.to(device),
            actions=self.actions.to(device),
            rewards=self.rewards.to(device),
            ends=self.ends.to(device),
            valid=self.valid.to(device),
            policy_logits=self.policy_logits.to(device),
            obs_values=self.obs_values.to(device),
        )


def rollout(
    actor_critic_model: ActorCriticModel,
    reward_end_model: RewardEndModel,
    next_obs_model: NextObsModel,
    obs_ctx_stk_init: torch.Tensor,
    action_seq_init: torch.Tensor,
    num_rollout_actions: int,
) -> Trajectory:
    batch_size, ctx_frames = obs_ctx_stk_init.shape[0:2]
    device = obs_ctx_stk_init.device

    actor_lstm_hidden: tuple[torch.Tensor, torch.Tensor] | None = None

    world_stepper = BatchedWorldStepper(
        ctx_frames,
        next_obs_model,
        reward_end_model,
        obs_ctx_stk_init,
        action_seq_init,
    )

    # Information we keep track of (assume we always start off valid)
    valid_all = torch.ones(
        (batch_size, num_rollout_actions), dtype=torch.bool, device=device
    )

    obs_last = obs_ctx_stk_init[:, -1, ...]

    # Trajectory information
    obs_list: list[torch.Tensor] = []
    actions_list: list[torch.Tensor] = []
    rewards_list: list[torch.Tensor] = []
    ends_list: list[torch.Tensor] = []
    policy_logits_list: list[torch.Tensor] = []
    values_list: list[torch.Tensor] = []

    # Get hidden for lstm (ig gradient flow here is up to us)
    with torch.no_grad():
        for idx in range(obs_ctx_stk_init.shape[1] - 1):
            _, _, actor_lstm_hidden = actor_critic_model(
                obs_ctx_stk_init[:, idx, ...], actor_lstm_hidden
            )

    for step in range(num_rollout_actions):
        # Policy and value at current observation
        policy_logits, values, actor_lstm_hidden = actor_critic_model(
            obs_last, actor_lstm_hidden
        )

        # Sample from the policies
        actions = Categorical(logits=policy_logits).sample()

        # Move in our world model
        with torch.no_grad():
            obs_next, reward_logits, end_logits = world_stepper.step(actions)

        # For rewards map [0, 1, 2] to [-1, 0, +1] by subtracting 1
        rewards = Categorical(logits=reward_logits).sample() - 1.0
        ends = Categorical(logits=end_logits).sample().bool()

        # Update and track info
        obs_list.append(obs_last)
        actions_list.append(actions)
        rewards_list.append(rewards)
        ends_list.append(ends)
        policy_logits_list.append(policy_logits)
        values_list.append(values)

        if step + 1 < num_rollout_actions:
            valid_all[:, step + 1] = (
                valid_all[:, step + 1] & valid_all[:, step] & (~ends)
            )

        obs_last = obs_next

    # Final frame with no corresponding actions
    obs_list.append(obs_last)
    with torch.no_grad():
        _, value_last, actor_lstm_hidden = actor_critic_model(
            obs_last, actor_lstm_hidden
        )
    values_list.append(value_last)

    return Trajectory(
        obs=torch.stack(obs_list, dim=1),
        actions=torch.stack(actions_list, dim=1),
        rewards=torch.stack(rewards_list, dim=1),
        ends=torch.stack(ends_list, dim=1),
        valid=valid_all,
        policy_logits=torch.stack(policy_logits_list, dim=1),
        obs_values=torch.stack(values_list, dim=1),
    )
