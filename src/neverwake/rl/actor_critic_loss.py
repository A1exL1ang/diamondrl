import torch
import torch.nn.functional as F

from neverwake.rl.actor_critic import ActorCriticModel
from neverwake.rl.rollout import Trajectory


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


def actor_critic_loss(
    actor_critic: ActorCriticModel,
    traj: Trajectory,
    weight_value: float,
    weight_entropy: float,
    lamb: float,
    gamma: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Returns are measured in units of their own running range, for the critic and for
    the policy (see ActorCriticModel for the range and where it comes from). The
    critic then never has to output a large number and the policy update keeps the
    same size, whatever the size of the rewards
    """
    # (B, H+1) the critic's outputs as actual returns
    values = actor_critic.to_return(traj.values)

    lamb_ret = lambda_returns(values, traj.rewards, traj.ends, lamb, gamma)

    num_valid = traj.valid.sum().clamp_min(1.0)
    valid_mask = traj.valid.float()

    probs_all_act = F.softmax(traj.policy_logits, dim=-1)
    log_probs_all_act = F.log_softmax(traj.policy_logits, dim=-1)

    # (B, H, A) gather across A
    log_probs = log_probs_all_act.gather(
        dim=2, index=traj.actions.unsqueeze(2)
    ).squeeze(2)

    # Note: the range is updated before it is used, so it includes this batch
    low, scale = actor_critic.update_return_range(lamb_ret[traj.valid])

    # Losses
    advantage_detach = (lamb_ret - values[:, :-1]).detach() / scale
    loss_policy = -((log_probs * advantage_detach * valid_mask).sum() / num_valid)

    value_target = (lamb_ret - low) / scale
    loss_value = weight_value * (
        (((traj.values[:, :-1] - value_target) ** 2) * valid_mask).sum() / num_valid
    )

    average_entropy = (
        (-probs_all_act * log_probs_all_act).sum(dim=2) * valid_mask
    ).sum() / num_valid
    loss_entropy = -weight_entropy * average_entropy

    loss = loss_policy + loss_value + loss_entropy

    return loss, {
        "loss_policy": loss_policy.detach(),
        "loss_value": loss_value.detach(),
        "loss_entropy": loss_entropy.detach(),
        "entropy": average_entropy.detach(),
        "valid_fraction": valid_mask.mean().detach(),
        "lamb_ret_mean": ((lamb_ret * valid_mask).sum() / num_valid).detach(),
        "return_low": low,
        "return_scale": scale,
    }
