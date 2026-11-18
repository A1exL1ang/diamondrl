from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch


class AbstractEnv(ABC):
    """
    Interface for environments.
    """

    @property
    @abstractmethod
    def all_actions(self) -> torch.Tensor:
        """
        Returns every available action (we can make the assumption of full
        legality here. Should not change.

        Returns:
            Every possible action regardless of legality
        """

    @property
    @abstractmethod
    def obs_shape(self) -> tuple[int, int, int]:
        """
        Returns the shape of the observation. Should not change.

        Returns:
            Size of observation
        """

    @property
    def achievements(self) -> tuple[str, ...]:
        """
        Returns the names of the achievements a game can unlock. An env that has any
        must list the ones unlocked so far in info["unlocked"] on every step, and
        define achievement_score.

        Returns:
            Every achievement (none unless the env overrides this)
        """
        return ()

    def achievement_score(self, success_rates: dict[str, float]) -> float:
        """
        Sums up how often each achievement is unlocked in one number, the way this
        env's benchmark defines it.

        Args:
            success_rates: For every achievement, the percentage of games that
                unlocked it.

        Returns:
            The score
        """
        raise NotImplementedError

    @abstractmethod
    def step(
        self, action: int
    ) -> tuple[torch.Tensor, float, bool, bool, dict[str, Any]]:
        """
        Advance the environment by one action.

        Args:
            action: Action to perform.

        Returns:
            observation: The observation after applying the action.
            reward: The immediate reward from this transition.
            terminated: Whether the episode ended due to a terminal state.
            truncated: Whether the episode ended due to a time/resource limit.
            info: Additional debugging or logging information.

        info["game_end"] must return the true value of whether the game ended or not
        """

    @abstractmethod
    def reset(self, seed: int | None) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        Resets the environment to an initial (potentially random) state.

        Args:
            seed: The seed to use for random initialization (if applicable)
        Returns:
            observation: Observation of the new initial environment
            info: Additional debugging or logging information
        """

    @abstractmethod
    def render(self) -> np.ndarray:
        """
        Renders the environment.

        Returns:
            Some form of rendering for the current state of the environment
        """

    @abstractmethod
    def close(self) -> None:
        """
        Cleans up the environment once the user finishes using it.
        """
