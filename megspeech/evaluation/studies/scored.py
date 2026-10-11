"""One model's probabilities on the scored words, read target by target and domain by domain."""

from dataclasses import dataclass

import numpy as np
import torch
from jaxtyping import Bool, Float, Int

from megspeech.data.labels.vocabulary import VOCAB_SIZE
from megspeech.evaluation.metric.ruler import Ruler


@dataclass(frozen=True)
class Scored:
    """One model's probabilities on the scored words, their labels and the domain each word is from."""

    labels: Int[np.ndarray, " n"]
    probs: Float[np.ndarray, "n v"]
    domains: np.ndarray

    def top1(self) -> Int[np.ndarray, " n"]:
        return self.probs.argmax(axis=1)

    def hits(self) -> Bool[np.ndarray, " n"]:
        """Whether the true word ranks in the top `TOP`, as the ruler counts it."""
        return Ruler().hits(torch.from_numpy(self.probs), torch.from_numpy(self.labels)).numpy()

    def support(self) -> Int[np.ndarray, " v"]:
        return np.bincount(self.labels, minlength=VOCAB_SIZE)

    def confusion(self) -> Int[np.ndarray, "v v"]:
        """Count of words with true label row predicted as column at top-1."""
        counts = np.zeros((VOCAB_SIZE, VOCAB_SIZE), dtype=np.int64)
        np.add.at(counts, (self.labels, self.top1()), 1)
        return counts

    def recall(self) -> Float[np.ndarray, " v"]:
        """Top-1 recall per target, NaN for a target with no word."""
        return self._rate(self.top1() == self.labels)

    def hit_rate(self) -> Float[np.ndarray, " v"]:
        """Top-`TOP` hit rate per target, NaN for a target with no word."""
        return self._rate(self.hits())

    def _rate(self, flags: Bool[np.ndarray, " n"]) -> Float[np.ndarray, " v"]:
        got = np.bincount(self.labels, weights=flags, minlength=VOCAB_SIZE)
        return np.where(self.support() > 0, got / np.maximum(self.support(), 1), np.nan)

    def macro(self) -> dict[str, float]:
        """Each domain's balanced top-`TOP` accuracy, and their mean: the selection ruler's number."""
        flags = self.hits()
        scores = {}
        for name in dict.fromkeys(self.domains):
            here = self.domains == name
            got = np.bincount(self.labels[here], weights=flags[here], minlength=VOCAB_SIZE)
            held = np.bincount(self.labels[here], minlength=VOCAB_SIZE)
            scores[str(name)] = float(np.mean(got[held > 0] / held[held > 0]))
        scores["macro"] = float(np.mean(list(scores.values())))
        return scores
