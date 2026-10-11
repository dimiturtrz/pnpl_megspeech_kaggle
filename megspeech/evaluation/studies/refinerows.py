"""The refiner's inputs: which rows of a span show their true state, batches of spans, the out-of-fold belief bank
external text's spans draw from, and real rows with the refiner's view of them."""

from dataclasses import dataclass, replace
from typing import Protocol

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
from torch.nn import functional

from megspeech.data.labels.vocabulary import IGNORE, VOCAB_SIZE
from megspeech.data.spans.spans import MAX_WORDS, PAD
from megspeech.evaluation.studies.boosted import LOG_FLOOR
from megspeech.evaluation.studies.refiner import STATES, Refiner
from megspeech.evaluation.studies.wikispans import OOV

CHUNK = 4096
"""Spans per forward when refining a whole section."""


class Showing(Protocol):
    """Which rows of a span show their true state instead of a belief."""

    name: str

    def drawn(self, states: Int[Tensor, "b w"], generator: torch.Generator) -> Bool[Tensor, "b w"]: ...

    def views(self, states: Int[Tensor, "g w"]) -> tuple[Int[Tensor, " k"], Bool[Tensor, "k w"]]: ...


class Hidden:
    """No row shows its state: the realistic refiner."""

    name = "hidden"

    def drawn(self, states: Int[Tensor, "b w"], generator: torch.Generator) -> Bool[Tensor, "b w"]:
        return torch.zeros(states.shape, dtype=torch.bool)

    def views(self, states: Int[Tensor, "g w"]) -> tuple[Int[Tensor, " k"], Bool[Tensor, "k w"]]:
        """Each span once."""
        return torch.arange(len(states)), torch.zeros(states.shape, dtype=torch.bool)


class Oracle:
    """Every row of a span but one shows its true state; the one left is graded."""

    name = "oracle"

    def drawn(self, states: Int[Tensor, "b w"], generator: torch.Generator) -> Bool[Tensor, "b w"]:
        real = states != PAD
        target = torch.rand(states.shape, generator=generator).masked_fill(~real, -1.0).argmax(1)
        return real & (torch.arange(states.shape[1]) != target[:, None])

    def views(self, states: Int[Tensor, "g w"]) -> tuple[Int[Tensor, " k"], Bool[Tensor, "k w"]]:
        """Each span once per row, that row hidden."""
        real = states != PAD
        spans, slots = real.nonzero(as_tuple=True)
        shown = real[spans] & (torch.arange(states.shape[1]) != slots[:, None])
        return spans, shown


@dataclass(frozen=True)
class Batch:
    """Spans of beliefs (log-probabilities), true states (`PAD` past the end) and the rows that show their state."""

    beliefs: Float[Tensor, "b w v"]
    states: Int[Tensor, "b w"]
    shown: Bool[Tensor, "b w"]

    def inputs(self) -> Float[Tensor, "b w f"]:
        safe = self.states.clamp(min=0).long()
        words = (safe < OOV)[..., None]
        oracle = torch.full(self.beliefs.shape, LOG_FLOOR).scatter(-1, safe.clamp(max=OOV - 1)[..., None], 0.0)
        oracle = torch.where(words, oracle, LOG_FLOOR)
        own = torch.where(self.shown[..., None], oracle, self.beliefs.clamp(min=LOG_FLOOR))
        flags = torch.stack([self.shown & (safe == OOV), self.shown], dim=-1).float()
        return torch.cat([own, flags], dim=-1)

    def targets(self) -> Int[Tensor, "b w"]:
        """Each graded row's state; `IGNORE` where a row shows it or is padding."""
        graded = (self.states != PAD) & ~self.shown
        return torch.where(graded, self.states.long(), IGNORE)

    def padding(self) -> Bool[Tensor, "b w"]:
        return self.states == PAD


@dataclass(frozen=True)
class Bank:
    """Real out-of-fold belief rows sorted by true state, each state's run `starts[s]` long `counts[s]`."""

    beliefs: Float[Tensor, "n v"]
    starts: Int[Tensor, " s"]
    counts: Int[Tensor, " s"]

    @classmethod
    def of(cls, beliefs: Float[Tensor, "n v"], labels: Int[Tensor, " n"]) -> "Bank":
        states = torch.where(labels >= 0, labels, OOV)
        counts = torch.bincount(states, minlength=STATES)
        if bool((counts == 0).any()):
            raise ValueError("a state has no out-of-fold row to draw")
        return cls(beliefs[states.argsort(stable=True)], counts.cumsum(0) - counts, counts)

    def draw(self, states: Int[Tensor, "b w"], generator: torch.Generator) -> Float[Tensor, "b w v"]:
        """A uniformly drawn row of each position's state; zeros past a span's end."""
        safe = states.clamp(min=0).long()
        offset = (torch.rand(states.shape, generator=generator) * self.counts[safe]).long()
        rows = self.beliefs[self.starts[safe] + offset]
        return torch.where((states != PAD)[..., None], rows, 0.0)


@dataclass(frozen=True)
class Simulated:
    """External text's spans of states, each belief drawn from the bank."""

    spans: Int[Tensor, "g w"]
    bank: Bank

    def count(self) -> int:
        return len(self.spans)

    def batch(self, spans: Int[Tensor, " b"], showing: Showing, generator: torch.Generator) -> Batch:
        states = self.spans[spans].long()
        return Batch(self.bank.draw(states, generator), states, showing.drawn(states, generator))

    def held(self, count: int, generator: torch.Generator) -> tuple["Simulated", "Real"]:
        """The spans but the last `count`, and those last ones with their beliefs drawn once, to stop stage 1 on."""
        states = self.spans[-count:].long()
        beliefs = self.bank.draw(states, generator)
        real = states != PAD
        index = torch.full(states.shape, PAD)
        index[real] = torch.arange(int(real.sum()))
        return replace(self, spans=self.spans[:-count]), Real(beliefs[real], states[real], index)


@dataclass(frozen=True)
class Real:
    """Real rows: each one's belief and state (`OOV` outside the vocabulary), and spans of row indices."""

    beliefs: Float[Tensor, "n v"]
    states: Int[Tensor, " n"]
    index: Int[Tensor, "g w"]

    @classmethod
    def of(cls, beliefs: Float[Tensor, "n v"], labels: Int[Tensor, " n"], index: Int[Tensor, "g w"]) -> "Real":
        return cls(beliefs.float().log_softmax(-1), torch.where(labels >= 0, labels, OOV), index)

    @staticmethod
    def chains(near: Int[Tensor, "n 2"]) -> Int[Tensor, "g w"]:
        """Spans walked from each row with no previous row along the next-row links."""
        starts = (near[:, 0] < 0).nonzero().squeeze(1)
        index = torch.full((len(starts), MAX_WORDS), PAD)
        index[:, 0] = starts
        for slot in range(1, MAX_WORDS):
            before = index[:, slot - 1]
            index[:, slot] = torch.where(before != PAD, near[before.clamp(min=0), 1], PAD)
        return index

    def take(self, keep: Bool[Tensor, " g"]) -> "Real":
        return replace(self, index=self.index[keep])

    def count(self) -> int:
        return len(self.index)

    def span_states(self, spans: Int[Tensor, " b"]) -> Int[Tensor, "b w"]:
        index = self.index[spans]
        return torch.where(index != PAD, self.states[index.clamp(min=0)], PAD)

    def batch(self, spans: Int[Tensor, " b"], showing: Showing, generator: torch.Generator) -> Batch:
        states = self.span_states(spans)
        beliefs = torch.where((states != PAD)[..., None], self.beliefs[self.index[spans].clamp(min=0)], 0.0)
        return Batch(beliefs, states, showing.drawn(states, generator))

    def refined(self, model: Refiner, showing: Showing) -> Float[Tensor, "n v"]:
        """Every row's refined log-probabilities over the 50, each graded in the views `showing` leaves it hidden; a
        row in no span keeps its belief."""
        out = self.beliefs.clone()
        spans, shown = showing.views(self.span_states(torch.arange(self.count())))
        model.eval()
        with torch.no_grad():
            for start in range(0, len(spans), CHUNK):
                chunk = spans[start : start + CHUNK]
                batch = replace(self.batch(chunk, Hidden(), torch.Generator()), shown=shown[start : start + CHUNK])
                device = next(model.parameters()).device
                logits = model(batch.inputs().to(device), batch.padding().to(device))[..., :VOCAB_SIZE]
                graded = ~batch.padding() & ~batch.shown
                out[self.index[chunk][graded]] = logits.log_softmax(-1).cpu()[graded]
        return out

    def loss(self, model: Refiner, showing: Showing) -> float:
        """Mean cross-entropy over the 50 of the in-vocabulary rows in spans."""
        refined = self.refined(model, showing)
        rows = self.index[self.index != PAD]
        rows = rows[self.states[rows] < OOV]
        return float(functional.nll_loss(refined[rows], self.states[rows]))
