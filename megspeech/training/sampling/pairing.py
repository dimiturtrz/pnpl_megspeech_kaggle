"""Co-batched cross-subject pairs for the instance term (megspeech-7gf.147.2): a natural draw of 64 spans holds a MASC
span and another subject's hearing of it well under once a batch, too rarely for the term to see positives, so a share
of each batch is drawn as pairs.

A span's key is the MASC token instance of its first word (`megspeech.data.labels.instances`): under sentence spans
every subject's hearing of a sentence is the same words, so the spans sharing a key hear the same sentence. The spans
are sorted by (key, subject); a span's partners are its key's block less its own subject's, drawn uniformly without a
host sync by skipping over that block. With `SamplerConfig.hearers` k > 2 an anchor's k - 1 partners sit at evenly
strided offsets from one uniform draw within that block, so they are distinct whenever it holds k - 1 spans
(megspeech-ct5).
"""

from dataclasses import dataclass
from typing import override

import numpy as np
import pandas as pd
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from megspeech.data.batching.split import Split
from megspeech.data.labels.instances import NO_INSTANCE, Instances
from megspeech.training.sampling.sampler import SamplerConfig


@dataclass(frozen=True)
class Unpaired:
    """Every span drawn alone: the draws as the curriculum made them, no random number drawn."""

    def drawn(self, draws: Int[Tensor, " d"], weights: Float[Tensor, " g"], batch: int) -> Int[Tensor, " d"]:
        """`draws`, in batches of `batch` spans drawn at `weights`, with this sampler's pairs written in."""
        return draws


@dataclass(frozen=True)
class Pairing(Unpaired):
    """`anchors` spans of each batch drawn among the `paired` ones, then `hearers - 1` partners of each. Per span, its
    key's block (`start`, `size`) and its own subject's block within it (`own`, `mine`) in the spans' (key, subject)
    `order`."""

    anchors: int
    hearers: int
    order: Int[Tensor, " g"]
    start: Int[Tensor, " g"]
    size: Int[Tensor, " g"]
    own: Int[Tensor, " g"]
    mine: Int[Tensor, " g"]
    paired: Bool[Tensor, " g"]

    @staticmethod
    def of(sampler: SamplerConfig, words: pd.DataFrame, train: Split, batch: int) -> Unpaired:
        """Pairs over `sampler.pairs` of each `batch` spans of `train`, whose words are `words` (`Corpus.positions`),
        `sampler.hearers` hearings a sentence."""
        anchors = round(sampler.pairs * batch / sampler.hearers)
        if not anchors:
            return Unpaired()
        instances = Instances.of(words)
        first = train.index[:, 0].cpu().numpy()
        key, subject = instances.code[first], instances.subject[first]
        order = np.lexsort((subject, key))
        start, size = Pairing.blocks(key[order][:, None], order)
        own, mine = Pairing.blocks(np.stack([key, subject], axis=1)[order], order)
        paired = (key != NO_INSTANCE) & (size > mine)
        if not paired.any():
            raise ValueError("no training span has another subject's hearing to pair with")
        device = train.index.device
        tensors = (torch.from_numpy(a).to(device) for a in (order, start, size, own, mine, paired))
        return Pairing(anchors, sampler.hearers, *tensors)

    @staticmethod
    def blocks(
        sorted_keys: Int[np.ndarray, "g k"], order: Int[np.ndarray, " g"]
    ) -> tuple[Int[np.ndarray, " g"], Int[np.ndarray, " g"]]:
        """Each span's block of equal `sorted_keys` rows (the spans' keys in `order`): its first place, its size."""
        _, first, inverse, counts = np.unique(
            sorted_keys, axis=0, return_index=True, return_inverse=True, return_counts=True
        )
        inverse = inverse.reshape(-1)
        start, size = np.empty_like(order), np.empty_like(order)
        start[order], size[order] = first[inverse], counts[inverse]
        return start, size

    def partners(self, spans: Int[Tensor, " a"]) -> Int[Tensor, "a k"]:
        """`hearers - 1` other subjects' hearings of each of the `paired` `spans`, each uniformly among them."""
        free = (self.size[spans] - self.mine[spans])[:, None]
        stride = (free // (self.hearers - 1)).clamp(min=1)
        steps = torch.arange(self.hearers - 1, device=spans.device)
        drawn = (torch.rand(spans.shape, device=spans.device)[:, None] * free).long()
        at = self.start[spans][:, None] + (drawn + steps * stride) % free
        at = at + self.mine[spans][:, None] * (at >= self.own[spans][:, None])
        return self.order[at]

    @override
    def drawn(self, draws: Int[Tensor, " d"], weights: Float[Tensor, " g"], batch: int) -> Int[Tensor, " d"]:
        batches = draws.view(-1, batch).clone()
        n, a = len(batches), self.anchors
        anchors = torch.multinomial(weights * self.paired, n * a, replacement=True)
        batches[:, :a] = anchors.view(n, a)
        batches[:, a : a * self.hearers] = self.partners(anchors).view(n, a, -1).transpose(1, 2).flatten(1)
        return batches.flatten()
