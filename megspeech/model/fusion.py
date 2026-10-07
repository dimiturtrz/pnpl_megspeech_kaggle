"""Feature-level fusion of word encoders trained apart (megspeech-7gf.131): each extra stem reads every window through
its own encoder, its words' features are projected to the transformer's width, and the projections are summed.

A sum of per-stem projections is a concatenation of the stems' features through one projection of the whole, with each
stem's block of it separately gateable; a block-wise gate is what branch dropout and the per-branch ablation need."""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager

import torch
from jaxtyping import Float
from torch import Tensor, nn

from megspeech.model.branch import Branch

PRIMARY = "primary"
"""The name of the model's own encoder among its sources."""


class Fusion(nn.Module):
    """The model's own encoder and its branches as `n` sources of features, gated per span as one."""

    def __init__(self, branches: Sequence[Branch], rate: float) -> None:
        """`rate` is the chance, in training, that a source is dropped from a span; with no branches nothing is."""
        super().__init__()
        self.stems = nn.ModuleList(branches)
        self.branches = tuple(branches)
        self.rate = rate if branches else 0.0
        self.silent = ""

    def names(self) -> tuple[str, ...]:
        """The sources' names, the model's own first."""
        return PRIMARY, *(str(branch.name) for branch in self.branches)

    def gates(self, rows: int, device: torch.device) -> Float[Tensor, "g n"]:
        """Each span's weight on each source. In training a source is dropped with probability `rate`, a span never
        losing all, and the survivors scaled to `n` between them, so a source's expected weight is 1; otherwise every
        source weighs 1. The silenced source weighs 0 either way."""
        names = self.names()
        n = len(names)
        keep = torch.ones(rows, n, dtype=torch.bool, device=device)
        if self.training and self.rate:
            keep = torch.rand(rows, n, device=device) >= self.rate
            lone = nn.functional.one_hot(torch.randint(n, (rows,), device=device), n).bool()
            keep = keep | (lone & ~keep.any(dim=1, keepdim=True))
        gates = keep * n / keep.sum(dim=1, keepdim=True)
        return gates * torch.tensor([float(name != self.silent) for name in names], device=device)

    def mix(self, parts: Sequence[Float[Tensor, "g w s d"]]) -> Float[Tensor, "g w s d"]:
        """The sources' `parts` summed under each span's gates."""
        gates = self.gates(len(parts[0]), parts[0].device).to(parts[0].dtype)
        return (torch.stack(list(parts)) * gates.t()[:, :, None, None, None]).sum(0)

    @contextmanager
    def silenced(self, name: str) -> Iterator[None]:
        """Within the block the source called `name` weighs 0: what the others alone decode."""
        if name not in self.names():
            raise KeyError(f"no source {name!r} among {self.names()}")
        self.silent = name
        try:
            yield
        finally:
            self.silent = ""

    def encoders(self) -> tuple[list[nn.Parameter], ...]:
        """Each branch's stage-1 parameters, apart from its projection."""
        return tuple(branch.encoder() for branch in self.branches)
