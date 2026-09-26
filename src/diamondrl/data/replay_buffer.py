from collections import Counter

import torch

from diamondrl.data.episode import Episode, MaskedSequenceBatch, RolloutBatch


class ReplayBuffer:
    def __init__(self, sample_weights: tuple[float, ...]):
        self.episodes: dict[int, Episode] = {}
        self.sample_weights = sample_weights

    def update_episode(self, episode: Episode, id: int) -> None:
        self.episodes[id] = episode

    def get_probs(self, n: int) -> torch.Tensor:
        probs = torch.zeros(n)

        for w, chunk in zip(
            self.sample_weights,
            torch.tensor_split(torch.arange(n), len(self.sample_weights)),
            strict=True,
        ):
            if len(chunk):
                probs[chunk] = w / len(chunk)

        # May not sum to 1 if n is too small
        return probs / probs.sum()

    def sample_rollout(
        self, batch_size: int, ctx_frames: int, roll_actions: int
    ) -> RolloutBatch:

        # Note: dict is chronological
        eligible = [
            i
            for i, ep in self.episodes.items()
            if len(ep.actions) >= max(roll_actions, 1)
        ]

        if not eligible:
            raise ValueError("tried to sample but no elegible things to be sampled")

        probs = self.get_probs(len(eligible))

        sampled_indices = torch.multinomial(probs, batch_size, replacement=True)
        counts = Counter(sampled_indices.tolist())

        indiv_samples = [
            self.episodes[eligible[idx]].sample_rollout(count, ctx_frames, roll_actions)
            for idx, count in counts.items()
        ]

        return RolloutBatch.cat(indiv_samples)

    def sample_sequence(
        self, batch_size: int, seq_transition_count: int
    ) -> MaskedSequenceBatch:
        eligible = [i for i, ep in self.episodes.items() if len(ep.actions) >= 1]

        if not eligible:
            raise ValueError("tried to sample but no elegible things to be sampled")

        probs = self.get_probs(len(eligible))

        sampled_indices = torch.multinomial(probs, batch_size, replacement=True)
        counts = Counter(sampled_indices.tolist())

        indiv_samples = [
            self.episodes[eligible[idx]].sample_sequence(count, seq_transition_count)
            for idx, count in counts.items()
        ]

        return MaskedSequenceBatch.cat(indiv_samples)
