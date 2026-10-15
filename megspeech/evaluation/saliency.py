"""Integrated gradients (Sundararajan et al. 2017, captum) of a scored word's label log-probability on the raw MEG
windows of its span, against a zero window: where a trained model's evidence for a word sits, through any
differentiable feature front end. A magnitude says where the evidence sits, not that the sensor causes the word."""

from dataclasses import dataclass

import torch
from captum.attr import IntegratedGradients
from jaxtyping import Float, Int
from torch import Tensor, nn

from megspeech.data.batching.split import SpanInput, Split


@dataclass(frozen=True)
class Objective:
    """One span's context head, read as each of `positions`' label log-probability: what captum differentiates and
    ablates. `x` and `onset` carry a leading dimension of `n` copies of the span (captum's steps and ablated copies),
    cycling through the targets."""

    model: nn.Module
    slot: Int[Tensor, "w"]
    subject: Int[Tensor, "r"]
    positions: Int[Tensor, "b"]
    labels: Int[Tensor, "b"]

    def __call__(self, x: Float[Tensor, "n r c t"], onset: Float[Tensor, "n w"]) -> Float[Tensor, " n"]:
        n, rows = x.shape[:2]
        which = torch.arange(n, device=x.device)
        slot = torch.where(self.slot < rows, self.slot + which[:, None] * rows, n * rows)
        context, _ = self.model(SpanInput(x.flatten(0, 1), onset, slot, self.subject.repeat(n)))
        target = which % len(self.labels)
        return context.log_softmax(-1)[which, self.positions[target], self.labels[target]]


@dataclass(frozen=True)
class Reads:
    """Some scored `words` of one span: the model's inputs for it and where each target sits."""

    x: Float[Tensor, "r c t"]
    onset: Float[Tensor, "w"]
    objective: Objective
    own: Int[Tensor, "b"]
    """Each target's own window, a row of `x`."""
    words: Int[Tensor, "b"]

    @classmethod
    def of(cls, model: nn.Module, split: Split, span: int, words: Int[Tensor, "b"]) -> "Reads":
        batch = split.batch(torch.tensor([span], device=split.y.device), split.index.shape[1])
        positions = (batch.index[0, None, :] == words[:, None]).float().argmax(dim=1)
        objective = Objective(model, batch.slot[0], batch.subject, positions, split.y[words])
        return cls(batch.x, batch.onset[0], objective, batch.slot[0][positions], words)

    def copies(self, count: int) -> tuple[Float[Tensor, "n r c t"], Float[Tensor, "n w"]]:
        return self.x.expand(count, *self.x.shape).contiguous(), self.onset.expand(count, -1).contiguous()


@dataclass(frozen=True)
class Integrated:
    """`model`'s words attributed to their own MEG windows, `chunk` words of a span per captum call."""

    model: nn.Module
    steps: int = 16
    """Integration steps from the zero window to the real one."""
    chunk: int = 4

    @staticmethod
    def share(values: Float[Tensor, "... k"]) -> Float[Tensor, "... k"]:
        return values / values.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(values.dtype).tiny)

    @staticmethod
    def spans(split: Split, words: Int[Tensor, " k"]) -> dict[int, Int[Tensor, " b"]]:
        """The `words` of each span that reads them, by the span's first read."""
        found = (split.index[:, :, None] == words[None, None, :]).any(dim=1).float()
        first = found.argmax(dim=0)
        return {int(span): words[first == span] for span in first.unique()}

    def reads(self, split: Split, words: Int[Tensor, " k"]) -> list[Reads]:
        self.model.eval()
        return [
            Reads.of(self.model, split, span, part)
            for span, group in self.spans(split, words).items()
            for part in group.split(self.chunk)
        ]

    def magnitude(self, read: Reads) -> Float[Tensor, "b c t"]:
        """Each target's magnitude of attribution over its own window."""
        count = len(read.own)
        x, _ = read.copies(count)
        method = IntegratedGradients(lambda window: read.objective(window, read.onset.expand(len(window), -1)))
        attributed = method.attribute(x, baselines=torch.zeros_like(x), n_steps=self.steps)
        return attributed[torch.arange(count), read.own].abs().detach()

    def sensors(self, split: Split, words: Int[Tensor, " k"]) -> Float[Tensor, "k c"]:
        """Each of `words`' share of its attribution per sensor, summed over its window, rows in `words`' order."""
        reads = self.reads(split, words)
        shares = torch.cat([self.share(self.magnitude(read).sum(dim=-1)) for read in reads])
        found = torch.cat([read.words for read in reads])
        return shares[(words[:, None] == found[None, :]).float().argmax(dim=1)]
