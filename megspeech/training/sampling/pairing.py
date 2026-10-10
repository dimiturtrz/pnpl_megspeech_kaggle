"""Co-batched cross-subject pairs for the instance term (megspeech-7gf.147.2): a natural draw of 64 spans holds a MASC
span and another subject's hearing of it well under once a batch, too rarely for the term to see positives, so a share
of each batch is drawn as pairs.

A span's key is the MASC token instance of its first word (`megspeech.data.labels.instances`): under sentence spans
every subject's hearing of a sentence is the same words, so the spans sharing a key hear the same sentence. The spans
are sorted by (key, subject); a span's partners are its key's block less its own subject's, drawn uniformly without a
host sync by skipping over that block.
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


@dataclass(frozen=True)
class Unpaired:
    """Every span drawn alone: the draws as the curriculum made them, no random number drawn."""

    def drawn(self, draws: Int[Tensor, " d"], weights: Float[Tensor, " g"], batch: int) -> Int[Tensor, " d"]:
        """`draws`, in batches of `batch` spans drawn at `weights`, with this sampler's pairs written in."""
        return draws


@dataclass(frozen=True)
class Pairing(Unpaired):
    """`anchors` spans of each batch drawn among the `paired` ones, each followed by a partner. Per span, its key's
    block (`start`, `size`) and its own subject's block within it (`own`, `mine`) in the spans' (key, subject)
    `order`."""

    anchors: int
    order: Int[Tensor, " g"]
    start: Int[Tensor, " g"]
    size: Int[Tensor, " g"]
    own: Int[Tensor, " g"]
    mine: Int[Tensor, " g"]
    paired: Bool[Tensor, " g"]

    @staticmethod
    def of(share: float, words: pd.DataFrame, train: Split, batch: int) -> Unpaired:
        """Pairs over `share` of each `batch` spans of `train`, whose words are `words` (`Corpus.positions`)."""
        anchors = round(share * batch / 2)
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
        return Pairing(anchors, *tensors)

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

    def partners(self, spans: Int[Tensor, " a"]) -> Int[Tensor, " a"]:
        """One other subject's hearing of each of the `paired` `spans`, uniformly among them."""
        free = self.size[spans] - self.mine[spans]
        at = self.start[spans] + (torch.rand(spans.shape, device=spans.device) * free).long()
        at = at + self.mine[spans] * (at >= self.own[spans])
        return self.order[at]

    @override
    def drawn(self, draws: Int[Tensor, " d"], weights: Float[Tensor, " g"], batch: int) -> Int[Tensor, " d"]:
        batches = draws.view(-1, batch).clone()
        n, a = len(batches), self.anchors
        anchors = torch.multinomial(weights * self.paired, n * a, replacement=True)
        batches[:, :a] = anchors.view(n, a)
        batches[:, a : 2 * a] = self.partners(anchors).view(n, a)
        return batches.flatten()
