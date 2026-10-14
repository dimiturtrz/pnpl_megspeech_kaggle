"""What the templates study's template distances encode: word durations streamed in, and a representational
similarity analysis against duration, phoneme and semantic distances."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn

from megspeech.data.labels.phonemes import Pronunciation
from megspeech.data.labels.vocabulary import PRIMARY, VOCAB_SIZE, Vocabulary

PERMUTATIONS = 1000


@dataclass
class Durations:
    """Streamed per-word sums of log duration (onset gap to the next word of the sentence)."""

    total: Float[Tensor, " v"] = field(default_factory=lambda: torch.zeros(VOCAB_SIZE, dtype=torch.float64))
    counts: Float[Tensor, " v"] = field(default_factory=lambda: torch.zeros(VOCAB_SIZE, dtype=torch.float64))

    def add(self, labels: Int[Tensor, " n"], log_gap: Float[Tensor, " n"]) -> None:
        """Count in the words whose gap is known (a sentence's last word has none)."""
        known = ~log_gap.isnan()
        self.total.index_add_(0, labels[known], log_gap[known].double())
        self.counts += torch.bincount(labels[known], minlength=VOCAB_SIZE)

    def mean(self) -> Float[Tensor, " v"]:
        return (self.total / self.counts.clamp(min=1)).float()


@dataclass(frozen=True)
class Rsa:
    """A word-by-word distance matrix compared with another by Spearman correlation of their upper triangles."""

    permutations: int = PERMUTATIONS
    seed: int = 0

    @staticmethod
    def gap(value: Float[Tensor, " v"]) -> Float[Tensor, "v v"]:
        """Absolute difference of a per-word scalar."""
        return (value[:, None] - value[None, :]).abs()

    @staticmethod
    def phonetic(pronunciation: Pronunciation) -> Float[Tensor, "v v"]:
        """Positions of the first phonemes at which two vocabulary words differ."""
        spelled = pronunciation.phonemes(torch.from_numpy(Vocabulary.token(pd.Series(PRIMARY.words))))
        return (spelled[:, None, :] != spelled[None, :, :]).sum(dim=-1).float()

    @staticmethod
    def semantic(vectors: Float[Tensor, "v d"]) -> Float[Tensor, "v v"]:
        """Cosine distance of word vectors."""
        unit = nn.functional.normalize(vectors, dim=-1)
        return 1 - unit @ unit.T

    @staticmethod
    def ranks(values: Float[Tensor, " n"]) -> Float[Tensor, " n"]:
        return torch.from_numpy(pd.Series(values.numpy()).rank().to_numpy(dtype=np.float32))

    def correlate(
        self, model: Float[Tensor, "v v"], target: Float[Tensor, "v v"], words: Bool[Tensor, " v"]
    ) -> tuple[float, float, int]:
        """Spearman r of the two matrices over pairs of `words`, its two-sided permutation p over word relabellings
        of the model matrix, and the pairs."""
        kept = torch.nonzero(words).flatten().cpu()
        rows, cols = torch.triu_indices(len(kept), len(kept), offset=1)
        sub = model.cpu()[kept][:, kept]
        order = self.ranks(sub[rows, cols])
        rank = torch.zeros(len(kept), len(kept))
        rank[rows, cols] = order
        rank = rank + rank.T
        goal = self.ranks(target.cpu()[kept][:, kept][rows, cols])
        goal = nn.functional.normalize(goal - goal.mean(), dim=0)

        def pearson(vals: Float[Tensor, "... n"]) -> Float[Tensor, "..."]:
            return nn.functional.normalize(vals - vals.mean(dim=-1, keepdim=True), dim=-1) @ goal

        found = pearson(order)
        generator = torch.Generator().manual_seed(self.seed)
        perm = torch.rand(self.permutations, len(kept), generator=generator).argsort(dim=1)
        null = pearson(rank[perm[:, rows], perm[:, cols]])
        extreme = int((null.abs() >= found.abs()).sum())
        return float(found), (1 + extreme) / (1 + self.permutations), len(rows)
