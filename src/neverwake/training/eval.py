from dataclasses import dataclass

import torch
from torch.distributions import Categorical

from neverwake.env.base import AbstractEnv
from neverwake.rl.actor_critic import ActorCriticModel


@dataclass
class GameRecord:
    obs: torch.Tensor  # (T + 1, C, H, W)
    actions: torch.Tensor  # (T,)
    rewards: torch.Tensor  # (T,)
    probs: torch.Tensor  # (T, A)
    values: torch.Tensor  # (T,)
    finished: bool  # False if cut off by the time limit or max_steps


@dataclass
class EvalSummary:
    returns: list[float]
    lengths: list[int]
    finished: list[bool]


class Evaluator:
    """
    Plays full games on its own env
    """

    def __init__(
        self,
        env: AbstractEnv,
        actor_critic: ActorCriticModel,
        eps: float,
        max_steps_per_game: int,
        seed: int,
    ):
        self.env = env
        self.actor_critic = actor_critic
        self.eps = eps
        self.max_steps_per_game = max_steps_per_game

        # State
        self.env.reset(seed)  # Seeded here

    def play_game(self, max_steps: int) -> GameRecord:
        device = next(self.actor_critic.parameters()).device
        num_actions = self.env.all_actions.shape[0]

        obs, _ = self.env.reset(None)
        lstm_state = None

        # Information
        frames: list[torch.Tensor] = [obs]
        actions: list[int] = []
        rewards: list[float] = []
        probs: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        finished = False

        while len(actions) < max_steps:
            # Action
            with torch.no_grad():
                policy_logits, value, lstm_state = self.actor_critic(
                    obs.unsqueeze(0).to(device), lstm_state
                )
            policy_logits = policy_logits.squeeze(0)

            action = Categorical(logits=policy_logits).sample().item()
            if torch.rand(()).item() < self.eps:
                action = torch.randint(0, num_actions, ()).item()

            # Step and record
            obs, reward, terminated, truncated, info = self.env.step(int(action))

            frames.append(obs)
            actions.append(int(action))
            rewards.append(reward)
            probs.append(policy_logits.softmax(dim=-1).cpu())
            values.append(value.squeeze(0).cpu())

            if terminated or truncated:
                if not info["game_end"]:
                    raise ValueError("need an env whose episodes are full games")
                finished = terminated
                break

        return GameRecord(
            obs=torch.stack(frames),
            actions=torch.tensor(actions).long(),
            rewards=torch.tensor(rewards).float(),
            probs=torch.stack(probs),
            values=torch.stack(values),
            finished=finished,
        )

    def evaluate(self, num_games: int) -> EvalSummary:
        returns: list[float] = []
        lengths: list[int] = []
        finished: list[bool] = []

        # Free up per-game information right away
        for _ in range(num_games):
            record = self.play_game(self.max_steps_per_game)
            returns.append(float(record.rewards.sum()))
            lengths.append(len(record.actions))
            finished.append(record.finished)

        return EvalSummary(returns=returns, lengths=lengths, finished=finished)
