"""A context-shifted span sampler: `spancoupling`'s context cells without their rows' fingerprints (7gf.145.1).

`Correlated` draws each row from the real rows sharing its (left, own, right) states. A trigram cell holds a few rows,
so the refiner meets the same row always beside the same neighbours and can learn the row, not the leak: on the
watched fold the context-cell arms drift to CE 2.7-2.9 against the identity's 2.409, while the own-state arms stay
near it. `Shifted` keeps the leak and drops the fingerprint. It draws the row from its own state alone
(`Correlated.coupled`, the span copula kept) on one half of the training spans, and adds the context's mean
log-probability offset (`Offsets`), measured on the other half: each level's cell mean minus its parent's, weighted
`n / (n + kappa)` with the same empirical-Bayes `kappa` as `Cells`. The offset belongs to the context and is shared by
every row drawn into it, so nothing in it identifies a row.
"""

from dataclasses import dataclass

import torch
from jaxtyping import Float, Int
from torch import Tensor

from megspeech.data.spans.spans import PAD
from megspeech.evaluation.studies.refinerows import Real
from megspeech.evaluation.studies.spancoupling import Cells, Context, Correlated


@dataclass(frozen=True)
class Offsets:
    """One context level's cells and each cell's shrunken mean log-probability offset from its parent cell."""

    keys: Int[Tensor, " c"]
    offsets: Float[Tensor, "c v"]

    @classmethod
    def of(cls, keys: Int[Tensor, " n"], parent: Int[Tensor, " n"], beliefs: Float[Tensor, "n v"]) -> "Offsets":
        cell, of_row = torch.unique(keys, return_inverse=True)
        _, parent_row = torch.unique(parent, return_inverse=True)
        x = beliefs.double()
        n = torch.bincount(of_row, minlength=len(cell)).double()
        mean = torch.zeros(len(cell), x.shape[1], dtype=x.dtype).index_add_(0, of_row, x) / n[:, None]
        parent_n = torch.bincount(parent_row).double()
        parent_mean = torch.zeros(len(parent_n), x.shape[1], dtype=x.dtype).index_add_(0, parent_row, x)
        cell_parent = torch.zeros(len(cell), dtype=torch.long).scatter_(0, of_row, parent_row)
        weight = n / (n + Cells.shrinkage(keys, parent, beliefs))
        shift = weight[:, None] * (mean - (parent_mean / parent_n[:, None])[cell_parent])
        return cls(cell, shift.float())

    def at(self, keys: Int[Tensor, "b w"]) -> Float[Tensor, "b w v"]:
        """Each key's offset, zero where no measured row had that context."""
        at = torch.searchsorted(self.keys, keys).clamp(max=len(self.keys) - 1)
        return torch.where((self.keys[at] == keys)[..., None], self.offsets[at], 0.0)


@dataclass(frozen=True)
class Shifted:
    """Own-state rows with the span copula, shifted by their context's measured offsets: a `Drawer`."""

    rows: Correlated
    levels: tuple[Offsets, ...]

    @classmethod
    def of(cls, real: Real) -> "Shifted":
        """Rows from the even training spans, context offsets from the odd ones."""
        even = torch.arange(real.count()) % 2 == 0
        measured = real.take(~even)
        spans = measured.span_states(torch.arange(measured.count()))
        inside = measured.index != PAD
        beliefs = measured.beliefs[measured.index[inside]]
        triple, pair, own = (key[inside] for key in Context.of(spans).cells())
        levels = (Offsets.of(triple, pair, beliefs), Offsets.of(pair, own, beliefs))
        return cls(Correlated.coupled(real.take(even)), levels)

    def draw(self, states: Int[Tensor, "b w"], generator: torch.Generator) -> Float[Tensor, "b w v"]:
        """One belief per position, zeros past a span's end."""
        keys = Context.of(states).cells()
        shift = torch.stack([level.at(key) for level, key in zip(self.levels, keys, strict=False)]).sum(0)
        shifted = (self.rows.draw(states, generator) + shift).log_softmax(-1)
        return torch.where((states != PAD)[..., None], shifted, 0.0)
