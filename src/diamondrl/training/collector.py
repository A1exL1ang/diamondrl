from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from diamondrl.data.episode import EpisodeBuilder
from diamondrl.data.replay_buffer import ReplayBuffer
from diamondrl.env.base import AbstractEnv
from diamondrl.models.actor_critic_model import ActorCriticModel


class Collector:
    """
    Single data collector
    """

    def __init__(
        self,
        env: AbstractEnv,
        replay_buffer: ReplayBuffer,
        actor_critic: ActorCriticModel,
        eps: float,
        seed: int,
    ):
        self.env = env
        self.replay_buffer = replay_buffer
        self.actor_critic = actor_critic
        self.eps = eps

        # State
        self.obs, _ = self.env.reset(seed)  # Seeded here
        self.lstm_hidden = None
        self.episode_builder = EpisodeBuilder(self.obs)

        # Tracking
        self.total_steps = 0
        self.game_count = 0  # Episode is a life, game is an actual game
        self.episode_count = 0  # Finished only
        self.reward_pos, self.reward_neg = 0, 0

        self.game_length = defaultdict(int)
        self.game_reward_sum = defaultdict(float)
        self.game_entropy_sum = defaultdict(float)

    def collect(self, num_steps: int) -> None:
        device = next(self.actor_critic.parameters()).device
        num_actions = self.env.all_actions.shape[0]

        for _ in range(num_steps):
            # Action
            with torch.no_grad():
                policy_logits, _, self.lstm_hidden = self.actor_critic(
                    self.obs.unsqueeze(0).to(device), self.lstm_hidden
                )
            policy_logits = policy_logits.squeeze(0)

            action = Categorical(logits=policy_logits).sample().item()
            if torch.rand(()).item() < self.eps:
                action = torch.randint(0, num_actions, ()).item()

            # Step and log data
            self.obs, reward, terminated, truncated, info = self.env.step(int(action))

            self.episode_builder.push_info(
                action=int(action),
                reward=reward,
                observation=self.obs,
                termination=terminated,
            )

            # Update telemetry
            self.total_steps += 1
            self.game_length[self.game_count] += 1

            self.reward_pos += reward > 0
            self.reward_neg += reward < 0

            self.game_reward_sum[self.game_count] += reward
            self.game_entropy_sum[self.game_count] += (
                -(
                    F.log_softmax(policy_logits, dim=-1)
                    * F.softmax(policy_logits, dim=-1)
                )
                .sum()
                .item()
            )

            if terminated or truncated:
                self.replay_buffer.update_episode(
                    self.episode_builder.snapshot(), self.episode_count
                )
                self.episode_count += 1

                # Determine if the game actually ended
                self.game_count += info["game_end"]

                # Reset (make sure to not seed here so it carries from previous seed)
                self.obs, _ = self.env.reset(None)
                self.lstm_hidden = None
                self.episode_builder = EpisodeBuilder(self.obs)

        # Update in-progress episode
        # Note that even one without any actions is fine since we filter suitable
        # episodes within the reward data collection
        self.replay_buffer.update_episode(
            self.episode_builder.snapshot(), self.episode_count
        )
