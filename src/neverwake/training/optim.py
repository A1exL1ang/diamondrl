from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR


def decay_groups(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    """
    Only matrices and conv kernels are decayed. Biases, norm gains and lone scalars
    (all ndim < 2) are left alone, and so are embeddings and Mamba's A_log (the log
    of how fast each slot of its state forgets)
    """
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []

    for module in model.modules():
        for name, param in module.named_parameters(recurse=False):
            if param.ndim < 2 or isinstance(module, nn.Embedding) or name == "A_log":
                no_decay.append(param)
            else:
                decay.append(param)

    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


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

        self.opt = AdamW(decay_groups(model, weight_decay), lr=lr)
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
    world_model: ModelOptimizer
    actor_critic: ModelOptimizer
