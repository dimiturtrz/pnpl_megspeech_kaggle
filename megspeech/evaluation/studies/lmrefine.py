"""A learned refiner of a span's word beliefs, pretrained on external text with simulated MEG confusion (7gf.145.1).

`lmsoft`'s hand-built GPT-2 fusion could not use weak neighbour beliefs. Here a small transformer reads a span of
per-row beliefs (the base's 50 context-head log-probabilities, every row, out-of-vocabulary ones included, as the
Kaggle test gives them) with each row's place from either end, and returns refined logits over the 50 and `OOV`,
residual on the row's own log-probabilities so an untrained refiner is the base.

Stage 1 (`text`) pretrains on WikiText-103's spans (`wikispans`). Each word's belief is a real out-of-fold row of the
same true state drawn uniformly (`Bank`): the base's confusion conditioned on the label, its empirical law, with no
fitted parameter. Stage 2 fine-tunes on the real cross-fitted rows. Both read the peer protocol's out-of-fold dump
(`kf5t-f0..4`, each fold model read on the sessions it never trained on, subject 0's training sessions only); its
last fold is held out to pick every stopping point and learning rate, the other folds train and fill the bank. The
instrument arm (`meg-only`) is the same refiner without stage 1, so the arms' difference is the external text's.

The refiner's change, `refined - head`, is the other channel of `lmsoft`'s fused finish (`head + lam * change - tau *
log_prior`, `lam = 1` the refiner alone), cross-fit on validation halves against the tau-only finish and transferred
to the local test exactly as `lmsoft` does. `oracle` refiners (`Oracle`) see every neighbour's true state as a one-hot
(`OOV` flagged, its word unknown) and only the graded row's belief: the harness check that the mechanism can carry
`lmbound`'s gain. `rebased` grades the same refiners on the out-of-fold models' own full-data base (`peerbase-s0`) on
validation, the base the refiner trained under.

    uv run python -m megspeech.evaluation.studies.lmrefine m11xsp7hub6d2v3 --peer peerbase-s0
"""

import argparse
import copy
import json
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Protocol

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
from torch.nn import functional

from core.paths import DataRoot
from megspeech.data.labels.vocabulary import IGNORE, VOCAB_SIZE
from megspeech.data.spans.spans import MAX_WORDS, PAD
from megspeech.evaluation.scoring.probabilities import PROBS, Probabilities
from megspeech.evaluation.studies.boosted import LOG_FLOOR
from megspeech.evaluation.studies.crowding import TEST, VAL, Arm, Rows, Section
from megspeech.evaluation.studies.crowding import Study as Crossfit
from megspeech.evaluation.studies.lmrows import ROWS, Posteriors
from megspeech.evaluation.studies.lmsoft import FUSED
from megspeech.evaluation.studies.lmsoft import Study as Soft
from megspeech.evaluation.studies.peer import EPOCHS
from megspeech.evaluation.studies.peer import Dump as Peer
from megspeech.evaluation.studies.refiner import STATES, Refiner
from megspeech.evaluation.studies.wikispans import OOV, OUT, STUDY
from megspeech.model.ensemble import Ensemble

LOG = logging.getLogger(__name__)

BATCH = 512
LRS = (1e-3, 3e-4)
"""Learning rates each stage picks between on the watched fold."""
CHECKS = 25
HELD = 20000
"""External spans held out of stage 1 to pick its stopping point and learning rate on its own task."""
"""Watched-fold checks over one pretraining pass."""
CHUNK = 4096
"""Spans per forward when refining a whole section."""
GPU_BUDGET = 2.5
"""GPU memory a run may take, GB: the GPU is shared with training lanes."""
GIB = 2**30
BASES = ("base", "rebased")


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


@dataclass
class Trainer:
    """Trains a refiner under one showing, keeping the state best on the watched fold."""

    showing: Showing
    watched: Real
    seed: int = 0
    curve: list[float] = field(default_factory=list)

    def step(self, model: Refiner, optimiser: torch.optim.Optimizer, batch: Batch) -> float:
        model.train()
        device = next(model.parameters()).device
        logits = model(batch.inputs().to(device), batch.padding().to(device))
        targets = batch.targets().to(device)
        loss = functional.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=IGNORE)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        return loss.item()

    def train(
        self, model: Refiner, source: Simulated | Real, lr: float, steps: int, every: int
    ) -> tuple[Refiner, float]:
        """`model` after `steps` batches, as it stood at the watched-fold check that scored best, and that score."""
        generator = torch.Generator().manual_seed(self.seed)
        optimiser = torch.optim.AdamW(model.parameters(), lr=lr)
        best, kept = self.watched.loss(model, self.showing), copy.deepcopy(model.state_dict())
        self.curve = [best]
        for step in range(1, steps + 1):
            spans = torch.randint(source.count(), (BATCH,), generator=generator)
            self.step(model, optimiser, source.batch(spans, self.showing, generator))
            if step % every == 0 or step == steps:
                self.curve.append(self.watched.loss(model, self.showing))
                LOG.info("%s lr %g step %d/%d watched %.4f", self.showing.name, lr, step, steps, self.curve[-1])
                if self.curve[-1] < best:
                    best, kept = self.curve[-1], copy.deepcopy(model.state_dict())
        model.load_state_dict(kept)
        return model, best

    def picked(
        self, start: Refiner, source: Simulated | Real, steps: int, every: int
    ) -> tuple[Refiner, dict[str, object]]:
        """The best of `start` trained at each of `LRS`, by the watched fold."""
        fits = [self.train(copy.deepcopy(start), source, lr, steps, every) for lr in LRS]
        at = min(range(len(LRS)), key=lambda k: fits[k][1])
        return fits[at][0], {"lr": LRS[at], "watched": fits[at][1], "by_lr": [fit[1] for fit in fits]}


@dataclass(frozen=True)
class Graded:
    """One section's graded words, every row of each domain, and the graded words' rows among them."""

    section: Section
    rows: dict[str, Real]
    ids: dict[str, Int[Tensor, " n"]]

    def fused(self, refined: dict[str, Float[Tensor, "n v"]]) -> dict[str, Rows]:
        """Each domain's words with the refiner's change to its rows' beliefs as the other channel."""
        domains = self.section.domains
        return {
            name: Rows(words.head, (refined[name] - self.rows[name].beliefs)[self.ids[name]], words.labels)
            for name, words in domains.items()
        }

    def rebased(self, beliefs: dict[str, Float[Tensor, "n v"]]) -> "Graded":
        """The same words under another base's beliefs of the same rows."""
        rows = {name: replace(real, beliefs=beliefs[name].float().log_softmax(-1)) for name, real in self.rows.items()}
        graded = self.section.domains
        domains = {
            name: Rows(rows[name].beliefs[self.ids[name]], words.other, words.labels) for name, words in graded.items()
        }
        return Graded(replace(self.section, domains=domains), rows, self.ids)

    def refined(self, model: Refiner, showing: Showing) -> dict[str, Rows]:
        return self.fused({name: real.refined(model, showing) for name, real in self.rows.items()})

    @classmethod
    def load(  # devtools-ignore: test-mirror
        cls, directories: Sequence[Path], rows: Path, section: str
    ) -> tuple["Graded", dict[str, float]]:
        graded = Section.load(directories, section, {})
        saved = Probabilities.load(directories[0] / PROBS, section).domains
        ids = {name: saved[name].ids[saved[name].novel] for name in graded.domains}
        dumped = Posteriors.load(str(rows), section)
        agreement = {name: Soft.aligned(dumped[name], graded.domains[name], ids[name]) for name in graded.domains}
        real = {name: Real.of(dumped[name].head, dumped[name].labels, dumped[name].index) for name in graded.domains}
        return cls(graded, real, ids), agreement


@dataclass(frozen=True)
class Study:
    """Every refiner's fused finish against the tau-only finish, on each base."""

    val: dict[str, Graded]
    test: Graded

    def scored(self, model: Refiner, showing: Showing, seed: int) -> dict[str, object]:
        """The fused finish's cross-fit gain on validation under each base, and on the local test under the first."""
        arms = (Arm.all()[0], FUSED)
        fits = {base: graded.refined(model, showing) for base, graded in self.val.items()}
        crossfits = {
            base: Crossfit(Section(fits[base], {}, graded.section.log_prior), None) for base, graded in self.val.items()
        }
        result: dict[str, object] = {f"{VAL}/{base}": crossfits[base].crossfit(arms, fits[base], seed) for base in fits}
        first = BASES[0]
        result[f"{TEST}/{first}"] = crossfits[first].transfer(arms, fits[first], self.test.refined(model, showing))
        return result

    @staticmethod
    def rebased(peer: Path, graded: Graded) -> Graded:  # devtools-ignore: test-mirror
        """`graded`'s validation words under the peer's full-data base, after its rows are checked to be the same."""
        dump = Peer.load(peer)
        val, names = dump.val, dump.domains
        beliefs: dict[str, Tensor] = {}
        for group, name in enumerate(names):
            mine = val.group == group
            if name in graded.rows:
                if not torch.equal(torch.where(val.y[mine] >= 0, val.y[mine], OOV), graded.rows[name].states):
                    raise ValueError(f"the peer's rows of {name} differ from the dumped rows")
                beliefs[name] = val.base[mine]
        return graded.rebased(beliefs)

    @staticmethod
    def folds(peer: Path) -> tuple[Real, Real]:  # devtools-ignore: test-mirror
        """The out-of-fold rows' spans: the training folds', then the last (watched) fold's."""
        oof = Peer.load(peer).oof
        real = Real.of(oof.base, oof.y, Real.chains(oof.near))
        fold = oof.group[real.index[:, 0]]
        watched = fold == fold.max()
        return real.take(~watched), real.take(watched)

    @staticmethod
    def main() -> None:  # devtools-ignore: test-mirror
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("name", help="an ensemble under models/ with probs.pt per member and lmrows.pt")
        parser.add_argument("--peer", default="peerbase-s0", help="the peer dump whose out-of-fold rows train")
        parser.add_argument("--passes", type=float, default=1.0, help="passes over the external text in stage 1")
        parser.add_argument("--epochs", type=int, default=EPOCHS, help="most fine-tuning epochs, picked on the fold")
        parser.add_argument("--held", type=int, default=HELD, help="external spans stage 1 stops on")
        parser.add_argument("--device", default="cpu")
        parser.add_argument(
            "--gpu-budget", type=float, default=GPU_BUDGET, help="most GPU memory this run may take, GB"
        )
        parser.add_argument("--seed", type=int, default=0)
        args = parser.parse_args()
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
        torch.manual_seed(args.seed)
        device = torch.device(args.device)
        if device.type == "cuda":
            torch.cuda.set_per_process_memory_fraction(args.gpu_budget * GIB / torch.cuda.mem_get_info(device)[1])
        root = DataRoot.from_config()
        out = root.models() / args.name
        study = root.models() / STUDY
        study.mkdir(parents=True, exist_ok=True)
        directories = Ensemble.trained(root.models(), args.name)
        val, val_aligned = Graded.load(directories, out / ROWS, VAL)
        test, test_aligned = Graded.load(directories, out / ROWS, TEST)
        peer = Peer.path(root, args.peer)
        train, watched = Study.folds(peer)
        rows = train.index[train.index != PAD]
        drawn = Simulated(torch.load(study / OUT)["spans"], Bank.of(train.beliefs[rows], train.states[rows]))
        simulated, held = drawn.held(args.held, torch.Generator().manual_seed(args.seed))
        graders = Study({BASES[0]: val, BASES[1]: Study.rebased(peer, val)}, test)
        steps = math.ceil(args.passes * simulated.count() / BATCH)
        epoch = math.ceil(train.count() / BATCH)
        stem = f"{STUDY}-h{args.held}-p{args.passes:g}-e{args.epochs}-s{args.seed}"
        result: dict[str, object] = {"aligned": {VAL: val_aligned, TEST: test_aligned}, "spans": simulated.count()}
        for showing in (Hidden(), Oracle()):
            trainer = Trainer(showing, watched, args.seed)
            base = watched.loss(Refiner().to(device), showing)
            pretrainer = Trainer(showing, held, args.seed)
            text, text_fit = pretrainer.picked(Refiner().to(device), simulated, steps, max(1, steps // CHECKS))
            text_fit["real"] = watched.loss(text, showing)
            models = {
                "text-only": (text, text_fit),
                "text": trainer.picked(text, train, args.epochs * epoch, epoch),
                "meg-only": trainer.picked(Refiner().to(device), train, args.epochs * epoch, epoch),
            }
            for arm, (model, fit) in models.items():
                key = f"{showing.name}/{arm}"
                torch.save(model.state_dict(), study / f"{args.name}-{stem}-{showing.name}-{arm}.pt")
                result[key] = {"fit": fit, "watched-base": base, **graders.scored(model, showing, args.seed)}
                LOG.info("%s %s", key, json.dumps(result[key])[:2000])
                (out / f"{stem}.json").write_text(json.dumps(result, indent=1))
        LOG.info("wrote %s", out / f"{stem}.json")


if __name__ == "__main__":
    Study.main()
