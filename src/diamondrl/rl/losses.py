import torch
import torch.nn.functional as F

from diamondrl.rl.rollout import Trajectory


@torch.no_grad()
def lambda_returns(
    values: torch.Tensor,
    rewards: torch.Tensor,
    ends: torch.Tensor,
    lamb: float,
    gamma: float,
) -> torch.Tensor:
    """
    values: (B, H+1)
    rewards: (B, H)
    ends: (B, H)
    """
    _, horizon = rewards.shape
    lamb_ret = torch.zeros_like(rewards).float()

    for t in range(horizon - 1, -1, -1):
        lamb_ret[:, t] = rewards[:, t] + gamma * (1.0 - ends[:, t].float()) * (
            (1 - lamb) * values[:, t + 1]
            + lamb * (lamb_ret[:, t + 1] if t + 1 < horizon else values[:, t + 1])
        )

    return lamb_ret


def rl_loss(
    traj: Trajectory,
    weight_value: float,
    weight_entropy: float,
    lamb: float,
    gamma: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

    lamb_ret = lambda_returns(traj.obs_values, traj.rewards, traj.ends, lamb, gamma)

    num_valid = traj.valid.sum().clamp_min(1.0)
    valid_mask = traj.valid.float()

    probs_all_act = F.softmax(traj.policy_logits, dim=-1)
    log_probs_all_act = F.log_softmax(traj.policy_logits, dim=-1)

    # (B, H, A) gather across A
    probs = probs_all_act.gather(dim=2, index=traj.actions.unsqueeze(2)).squeeze(2)
    log_probs = log_probs_all_act.gather(
        dim=2, index=traj.actions.unsqueeze(2)
    ).squeeze(2)

    # Losses
    advantage_detach = (lamb_ret - traj.obs_values[:, :-1]).detach()
    loss_policy = -((log_probs * advantage_detach * valid_mask).sum() / num_valid)

    loss_value = weight_value * (
        (((traj.obs_values[:, :-1] - lamb_ret) ** 2) * valid_mask).sum() / num_valid
    )

    average_entropy = (-probs * log_probs * valid_mask).sum() / num_valid
    loss_entropy = -weight_entropy * average_entropy

    loss_total = loss_policy + loss_value + loss_entropy

    return loss_total, {
        "loss_policy": loss_policy.detach(),
        "loss_value": loss_value.detach(),
        "loss_entropy": loss_entropy.detach(),
        "entropy": average_entropy.detach(),
        "valid_fraction": valid_mask.mean().detach(),
        "lamb_ret_mean": ((lamb_ret * valid_mask).sum() / num_valid).detach(),
    }
