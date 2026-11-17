import torch
import torch.nn.functional as F

# A twohot head predicts a number by classifying over fixed bins and reading off the
# expected bin value. This is the symexp twohot head of DreamerV3 (Hafner et al. 2023,
# "Mastering Diverse Domains through World Models", arXiv:2301.04104). The bins are
# fine near 0 and coarser further out, so real game scores of any size fit. DreamerV3
# uses 255 bins with a limit of 20, which are about 0.17 apart near 0 and reach up to
# about +-485 million


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.expm1(torch.abs(x))


def twohot_bins(num_bins: int, limit: float) -> torch.Tensor:
    """
    num_bins: has to be odd, so that the middle bin is exactly 0
    limit: the bins are evenly spaced from -limit to +limit before symexp, so the
           largest one is e^limit - 1

    Returns the bin values (num_bins,)
    """
    if num_bins < 3 or num_bins % 2 == 0:
        raise ValueError("need an odd number of twohot bins (at least 3)")

    # Note: built as the negative half and its mirror, so bin i is exactly minus
    # bin num_bins - 1 - i and the middle one is exactly 0
    half = symexp(torch.linspace(-limit, 0.0, num_bins // 2 + 1))
    return torch.cat([half, -half[:-1].flip(0)])


def twohot_decode(logits: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """
    logits: (..., num_bins)
    bins: (num_bins,)

    Returns the expected bin value (...)
    """
    weighted = F.softmax(logits.float(), dim=-1) * bins

    # Note: every bin is added to its mirror before the sum, so equal probabilities
    # on 2 mirrored bins cancel exactly (a zero-init head predicts exactly 0). A
    # plain sum leaves a rounding error as large as the far bins
    half = bins.shape[0] // 2
    return (weighted[..., half + 1 :] + weighted[..., :half].flip(-1)).sum(dim=-1)


def twohot_loss(
    logits: torch.Tensor, target: torch.Tensor, bins: torch.Tensor
) -> torch.Tensor:
    """
    Cross entropy against the target split over its 2 nearest bins, with more weight
    on the closer one (so the expected bin value of the split is the target itself)

    logits: (..., num_bins)
    target: (...)
    bins: (num_bins,)

    Returns the loss of every entry (...)
    """
    target = target.float().clamp(min=bins[0], max=bins[-1])

    # Nearest bin at or below the target and the share of the one above it
    below = (torch.searchsorted(bins, target, right=True) - 1).clamp(
        min=0, max=bins.shape[0] - 2
    )
    share_above = (target - bins[below]) / (bins[below + 1] - bins[below])

    log_probs = F.log_softmax(logits.float(), dim=-1)
    log_prob_below = log_probs.gather(-1, below.unsqueeze(-1)).squeeze(-1)
    log_prob_above = log_probs.gather(-1, (below + 1).unsqueeze(-1)).squeeze(-1)

    return -((1.0 - share_above) * log_prob_below + share_above * log_prob_above)
