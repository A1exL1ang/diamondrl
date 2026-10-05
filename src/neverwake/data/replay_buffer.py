from collections import Counter

import torch

from neverwake.data.episode import ContextBatch, Episode, MaskedSequenceBatch


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

    def sample_context(
        self, batch_size: int, ctx_frames: int, future_frames: int
    ) -> ContextBatch:

        # Note: dict is chronological
        eligible = [
            i
            for i, ep in self.episodes.items()
            if len(ep.actions) >= max(future_frames, 1)
        ]

        if not eligible:
            raise ValueError("tried to sample but no elegible things to be sampled")

        probs = self.get_probs(len(eligible))

        sampled_indices = torch.multinomial(probs, batch_size, replacement=True)
        counts = Counter(sampled_indices.tolist())

        indiv_samples = [
            self.episodes[eligible[idx]].sample_context(
                count, ctx_frames, future_frames
            )
            for idx, count in counts.items()
        ]

        return ContextBatch.cat(indiv_samples)

    def sample_masked_sequence(
        self, batch_size: int, num_transitions: int
    ) -> MaskedSequenceBatch:
        eligible = [i for i, ep in self.episodes.items() if len(ep.actions) >= 1]

        if not eligible:
            raise ValueError("tried to sample but no elegible things to be sampled")

        probs = self.get_probs(len(eligible))

        sampled_indices = torch.multinomial(probs, batch_size, replacement=True)
        counts = Counter(sampled_indices.tolist())

        indiv_samples = [
            self.episodes[eligible[idx]].sample_masked_sequence(count, num_transitions)
            for idx, count in counts.items()
        ]

        return MaskedSequenceBatch.cat(indiv_samples)
