from dataclasses import dataclass

import torch


@dataclass
class TrajectorySample:
    obs_stacked: torch.Tensor  # (B, L, C, H, W)
    obs_next: torch.Tensor  # (B, C, H, W)
    actions: torch.Tensor  # (B,)
    rewards: torch.Tensor  # (B,)
    terminations: torch.Tensor  # (B,)


class TrajectoryStore:
    def __init__(
        self,
        train_steps: int,
        initial_observation: torch.Tensor,
    ) -> None:
        self.observations: list[torch.Tensor] = [initial_observation]
        self.actions: list[int] = []
        self.rewards: list[float] = []
        self.terminations: list[bool] = []

    def push_info(
        self, action: int, reward: float, observation: torch.Tensor, termination: bool
    ) -> None:
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.terminations.append(termination)

    def sample_info(self, num_samples: int, seq_length: int) -> TrajectorySample:
        n, stk = len(self.observations), seq_length

        if stk - 1 > n - 1:
            raise ValueError(f"have {n} obs and seq length of {stk}")

        indices = torch.randint(stk - 1, n - 1, (num_samples,))

        obs_stacked = torch.stack(
            [torch.stack(self.observations[i - stk + 1 : i + 1]) for i in indices]
        )
        obs_next = torch.stack([self.observations[i + 1] for i in indices])

        actions = torch.tensor([self.actions[i] for i in indices], dtype=torch.long)

        rewards = torch.tensor([self.rewards[i] for i in indices], dtype=torch.float)

        terminations = torch.tensor(
            [self.terminations[i] for i in indices], dtype=torch.long
        )

        return TrajectorySample(
            obs_stacked=obs_stacked,
            obs_next=obs_next,
            actions=actions,
            rewards=rewards,
            terminations=terminations,
        )
