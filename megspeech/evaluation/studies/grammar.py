"""A first-order part-of-speech grammar and the Viterbi decode of a sentence's tags under it."""

import itertools
from dataclasses import dataclass

import numpy as np
from jaxtyping import Float, Int


@dataclass(frozen=True)
class Grammar:
    """A first-order tag grammar: a sentence's first tag's log-probability and each next tag's given the one before,
    counted add-one on training sentences."""

    start: Float[np.ndarray, " k"]
    transition: Float[np.ndarray, "k k"]

    @classmethod
    def of(
        cls, tags: Int[np.ndarray, " n"], sentences: Int[np.ndarray, " n"], onsets: Float[np.ndarray, " n"], k: int
    ) -> "Grammar":
        """The grammar of `tags` (each below `k`) read in onset order within each sentence."""
        order = np.lexsort((onsets, sentences))
        tag, sentence = tags[order], sentences[order]
        follows = sentence[1:] == sentence[:-1]
        start = np.bincount(tag[np.r_[True, ~follows]], minlength=k) + 1.0
        pairs = np.ones((k, k))
        np.add.at(pairs, (tag[:-1][follows], tag[1:][follows]), 1)
        return cls(np.log(start / start.sum()), np.log(pairs / pairs.sum(1, keepdims=True)))

    def viterbi(self, emissions: Float[np.ndarray, "t k"]) -> Int[np.ndarray, " t"]:
        """The most probable tag path through one sentence's per-word emission log-likelihoods."""
        score = self.start + emissions[0]
        back = np.zeros(emissions.shape, np.int64)
        for t in range(1, len(emissions)):
            candidates = score[:, None] + self.transition
            back[t] = candidates.argmax(0)
            score = candidates.max(0) + emissions[t]
        path = [int(score.argmax())]
        for t in range(len(emissions) - 1, 0, -1):
            path.append(int(back[t, path[-1]]))
        return np.array(path[::-1])

    def decode(
        self, emissions: Float[np.ndarray, "n k"], sentences: Int[np.ndarray, " n"], onsets: Float[np.ndarray, " n"]
    ) -> Int[np.ndarray, " n"]:
        """Each sentence's words' tags decoded jointly, in onset order, returned in the rows' order."""
        order = np.lexsort((onsets, sentences))
        sentence = sentences[order]
        bounds = np.flatnonzero(np.r_[True, sentence[1:] != sentence[:-1], True])
        decoded = np.empty(len(order), np.int64)
        for begin, end in itertools.pairwise(bounds):
            rows = order[begin:end]
            decoded[rows] = self.viterbi(emissions[rows])
        return decoded
