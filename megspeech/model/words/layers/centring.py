"""Word embeddings centred on their own run's mean.

A run's offset dominates the word embeddings (run eta-squared .226 against the word's .078; mean pairwise cosine .78
before centring on the run, near 0 after), and a low-signal word's embedding sits at its run's centroid.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import override

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn

RUN_MOMENTUM = 0.99
"""Per-update decay of a run's training mean: a time constant of 100 updates of that run. A batch of 64 spans
across the training runs holds a given run every few steps with ~10 of its words, so the mean averages ~1000 words
(noise ~3% of the per-word spread) over a few hundred steps, short against an epoch's drift of the encoder."""


class RunCentring(nn.Module):
    """Each embedding less its run's mean, the mean held out of the gradient; inactive, the identity.

    Training reads an exponential moving mean per run, updated from each batch. Evaluation reads each run's mean over
    all its windows, gathered by a first sweep (`calibrated`): label-free, and the same rule for a run training never
    saw. Uncalibrated, evaluation reads the mean of each run's words in the batch. No state is saved with the model.
    """

    mean: Tensor
    seen: Tensor

    def __init__(self, width: int, *, active: bool, momentum: float = RUN_MOMENTUM) -> None:
        super().__init__()
        self.width = width
        self.active = active
        self.momentum = momentum
        self.register_buffer("mean", torch.zeros(0, width), persistent=False)
        self.register_buffer("seen", torch.zeros(0, dtype=torch.bool), persistent=False)
        self.fixed: Float[Tensor, "k e"] | None = None
        self.gathered: list[tuple[Float[Tensor, "k e"], Float[Tensor, " k"]]] | None = None

    def reserve(self, runs: int) -> None:
        """Room for the training means of `runs` runs, numbered from 0, each unseen."""
        self.mean = self.mean.new_zeros(runs, self.width)
        self.seen = self.seen.new_zeros(runs)

    @override
    def forward(
        self, slots: Float[Tensor, "g w e"], mask: Bool[Tensor, "g w"], run: Int[Tensor, "g w"]
    ) -> Float[Tensor, "g w e"]:
        """`slots` less each `mask`ed position's `run` mean; empty positions stay as they are."""
        if not self.active:
            return slots
        mean = self.centre(slots.detach().float(), mask, run.clamp(min=0))
        return slots - (mean[run.clamp(min=0)] * mask[..., None]).to(slots.dtype)

    def centre(
        self, slots: Float[Tensor, "g w e"], mask: Bool[Tensor, "g w"], run: Int[Tensor, "g w"]
    ) -> Float[Tensor, "k e"]:
        """Each run's mean for this batch, updating whatever state the mode keeps."""
        if self.gathered is not None:
            self.gathered.append(self.tally(slots, mask, run, self.runs(run)))
            return slots.new_zeros(self.runs(run), self.width)
        if self.training:
            return self.update(*self.tally(slots, mask, run, len(self.mean)))
        if self.fixed is not None:
            return self.fixed
        sums, counts = self.tally(slots, mask, run, self.runs(run))
        return sums / counts.clamp(min=1)[:, None]

    def update(self, sums: Float[Tensor, "k e"], counts: Float[Tensor, " k"]) -> Float[Tensor, "k e"]:
        """The training means after one batch's per-run `sums` over `counts` words: a run's first batch sets its mean,
        each later one moves it by `1 - momentum`; runs absent from the batch keep theirs."""
        present = counts > 0
        batch = sums / counts.clamp(min=1)[:, None]
        moved = torch.where(self.seen[:, None], self.momentum * self.mean + (1 - self.momentum) * batch, batch)
        self.mean.copy_(torch.where(present[:, None], moved, self.mean))
        self.seen |= present
        return self.mean

    @staticmethod
    def runs(run: Int[Tensor, "g w"]) -> int:
        """How many runs the numbering `run` reaches."""
        return int(run.max()) + 1 if run.numel() else 0

    @staticmethod
    def tally(
        slots: Float[Tensor, "g w e"], mask: Bool[Tensor, "g w"], run: Int[Tensor, "g w"], runs: int
    ) -> tuple[Float[Tensor, "k e"], Float[Tensor, " k"]]:
        """Each of `runs` runs' sum of its `mask`ed `slots`, and how many it summed."""
        weight = mask.flatten().to(slots.dtype)
        index = run.flatten()
        sums = slots.new_zeros(runs, slots.shape[-1]).index_add_(0, index, slots.flatten(0, 1) * weight[:, None])
        return sums, slots.new_zeros(runs).index_add_(0, index, weight)

    def collect(self) -> None:
        """Gather every run's sums from the batches that follow, centring none of them."""
        self.gathered = []

    def settle(self) -> None:
        """Fix each run's evaluation mean as the mean over every word `collect` gathered."""
        gathered = self.gathered or []
        runs = max((len(counts) for _, counts in gathered), default=0)
        sums = torch.zeros(runs, self.width, device=self.mean.device)
        counts = torch.zeros(runs, device=self.mean.device)
        for part, count in gathered:
            sums[: len(count)] += part
            counts[: len(count)] += count
        self.fixed = sums / counts.clamp(min=1)[:, None]
        self.gathered = None

    def release(self) -> None:
        """Forget the evaluation means, whose run numbering belongs to the split they were gathered on."""
        self.fixed = None
        self.gathered = None

    @classmethod
    @contextmanager
    def calibrated(cls, model: nn.Module, sweep: Callable[[], object]) -> Iterator[None]:
        """Within the block, each active centring of `model` reads the run means one `sweep` over the split gathers;
        with none active, `sweep` never runs."""
        centrings = [module for module in model.modules() if isinstance(module, cls) and module.active]
        for centring in centrings:
            centring.collect()
        if centrings:
            sweep()
        for centring in centrings:
            centring.settle()
        try:
            yield
        finally:
            for centring in centrings:
                centring.release()
