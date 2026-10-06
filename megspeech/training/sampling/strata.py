"""Which stratum each training span belongs to, for the samplers that balance or tilt by stratum."""

import torch
from jaxtyping import Float, Int
from torch import Tensor

from megspeech.data.batching.split import Split
from megspeech.data.spans.spans import PAD

BINS = 3
"""Quantile bins of span length and of word rate within a stratum: terciles, so each bin holds a third of the spans
and a subject's strata stay large enough to measure an epoch's loss on."""


class Strata:
    """Which stratum each span belongs to: its subject, its word count and its word rate."""

    @staticmethod
    def stratify(train: Split) -> Int[Tensor, " g"]:
        """Each span's stratum: its subject, then its word count's and its words per second's `BINS`-quantile bins
        (a one-word span's rate is 0)."""
        length = (train.index != PAD).sum(dim=1)
        last = train.onset.gather(1, (length - 1).clamp(min=0)[:, None])[:, 0]
        rate = torch.where(length > 1, (length - 1) / (last - train.onset[:, 0]), 0.0)
        subject = train.subject[train.index[:, 0]]
        return (subject * BINS + Strata.binned(length.float())) * BINS + Strata.binned(rate)

    @staticmethod
    def binned(values: Float[Tensor, " g"]) -> Int[Tensor, " g"]:
        """Each value's `BINS`-quantile bin, 0 the lowest."""
        edges = torch.quantile(values, torch.linspace(0, 1, BINS + 1, device=values.device)[1:-1])
        return torch.bucketize(values, edges)

    @staticmethod
    def sourced(train: Split) -> Int[Tensor, " g"]:
        """`stratify`'s strata split by each span's recording source, numbered from 0; as `stratify` where the source
        is unknown."""
        strata = Strata.stratify(train)
        if train.source is None:
            return strata
        source = train.source[train.index[:, 0]]
        return torch.unique(source * (int(strata.max()) + 1) + strata, return_inverse=True)[1]
