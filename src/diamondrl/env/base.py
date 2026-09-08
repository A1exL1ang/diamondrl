from abc import ABC, abstractmethod


class AbstractGame(ABC):
    @abstractmethod
    def step(self, action: int) -> int:
        pass
