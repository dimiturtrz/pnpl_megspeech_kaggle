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

    uv run python -m megspeech.evaluation.studies lm.refine m11xsp7hub6d2v3 --peer peerbase-s0
"""

import argparse
import copy
import functools
import json
import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.nn import functional

from core.paths import DataRoot
from megspeech.data.labels.vocabulary import IGNORE
from megspeech.evaluation.scoring.probabilities import PROBS, Probabilities
from megspeech.evaluation.studies.errors.crowding import TEST, VAL, Arm, Rows, Section
from megspeech.evaluation.studies.errors.crowding import Study as Crossfit
from megspeech.evaluation.studies.experts.peer import EPOCHS
from megspeech.evaluation.studies.experts.peer import Dump as Peer
from megspeech.evaluation.studies.lm.rows import ROWS, Posteriors
from megspeech.evaluation.studies.lm.soft import FUSED
from megspeech.evaluation.studies.lm.soft import Study as Soft
from megspeech.evaluation.studies.refiner.belief import Refiner
from megspeech.evaluation.studies.refiner.coupling import Correlated, Profile
from megspeech.evaluation.studies.refiner.drawer import Drawer
from megspeech.evaluation.studies.refiner.rows import Bank, Batch, Hidden, Oracle, Real, Showing, Simulated
from megspeech.evaluation.studies.refiner.shift import Shifted
from megspeech.evaluation.studies.refiner.wiki import OOV, OUT, STUDY
from megspeech.model.ensemble import Ensemble

LOG = logging.getLogger(__name__)

BATCH = 512
LRS = (1e-3, 3e-4)
"""Learning rates each stage picks between on the watched fold."""
CHECKS = 25
HELD = 20000
"""External spans held out of stage 1 to pick its stopping point and learning rate on its own task."""
"""Watched-fold checks over one pretraining pass."""
GPU_BUDGET = 2.5
"""GPU memory a run may take, GB: the GPU is shared with training lanes."""
GIB = 2**30
BASES = ("base", "rebased")
ARMS = ("text-only", "text", "meg-only")
"""Stage 1 alone, stage 1 then real fine-tuning, real fine-tuning alone."""
SAMPLERS: dict[str, Callable[[Real], Drawer]] = {
    "independent": Bank.spanned,
    "correlated": Correlated.of,
    "contextual": Correlated.contextual,
    "coupled": Correlated.coupled,
    "shifted": Shifted.of,
}
"""How a simulated span's beliefs are drawn from the training folds' rows (`spancoupling`)."""
SHOWINGS: dict[str, Callable[[], Showing]] = {"hidden": Hidden, "oracle": Oracle}


class Corpus:
    """The label sequences stage 1 simulates beliefs over."""

    @staticmethod
    def wiki(study: Path, train: Real) -> Int[Tensor, "g w"]:
        """WikiText-103's spans (`wikispans`)."""
        return torch.load(study / OUT)["spans"]

    @staticmethod
    def meg(study: Path, train: Real) -> Int[Tensor, "g w"]:
        """The training folds' own spans of states: the sampler alone differs from real fine-tuning."""
        return train.span_states(torch.arange(train.count()))


CORPORA: dict[str, Callable[[Path, Real], Int[Tensor, "g w"]]] = {"wiki": Corpus.wiki, "meg": Corpus.meg}


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
class PeerBase:
    """A peer dump: its out-of-fold rows train the refiner, its full-data base rebases validation."""

    path: Path

    def rebased(self, graded: Graded) -> Graded:  # devtools-ignore: test-mirror
        """`graded`'s validation words under the peer's full-data base, after its rows are checked to be the same."""
        dump = Peer.load(self.path)
        val, names = dump.val, dump.domains
        beliefs: dict[str, Tensor] = {}
        for group, name in enumerate(names):
            mine = val.group == group
            if name in graded.rows:
                if not torch.equal(torch.where(val.y[mine] >= 0, val.y[mine], OOV), graded.rows[name].states):
                    raise ValueError(f"the peer's rows of {name} differ from the dumped rows")
                beliefs[name] = val.base[mine]
        return graded.rebased(beliefs)

    def folds(self) -> tuple[Real, Real]:  # devtools-ignore: test-mirror
        """The out-of-fold rows' spans: the training folds', then the last (watched) fold's."""
        oof = Peer.load(self.path).oof
        real = Real.of(oof.base, oof.y, Real.chains(oof.near))
        fold = oof.group[real.index[:, 0]]
        watched = fold == fold.max()
        return real.take(~watched), real.take(watched)


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
    def parser() -> argparse.ArgumentParser:
        """The study's command line: which peer dump trains, which sampler and arms run."""
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("name", help="an ensemble under models/ with probs.pt per member and lmrows.pt")
        parser.add_argument("--peer", default="peerbase-s0", help="the peer dump whose out-of-fold rows train")
        parser.add_argument("--passes", type=float, default=1.0, help="passes over the external text in stage 1")
        parser.add_argument("--epochs", type=int, default=EPOCHS, help="most fine-tuning epochs, picked on the fold")
        parser.add_argument(
            "--held", type=int, default=HELD, help="simulated spans stage 1 stops on; 0: the watched fold"
        )
        parser.add_argument("--sampler", choices=SAMPLERS, default="independent")
        parser.add_argument("--corpus", choices=CORPORA, default="wiki")
        parser.add_argument("--showings", nargs="+", choices=SHOWINGS, default=list(SHOWINGS))
        parser.add_argument(
            "--arms", nargs="+", choices=ARMS, default=list(ARMS), help="meg-only: real out-of-fold rows alone"
        )
        parser.add_argument("--device", default="cpu")
        parser.add_argument(
            "--gpu-budget", type=float, default=GPU_BUDGET, help="most GPU memory this run may take, GB"
        )
        parser.add_argument("--seed", type=int, default=0)
        return parser

    @staticmethod
    def main() -> None:  # devtools-ignore: test-mirror
        args = Study.parser().parse_args()
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
        peer = PeerBase(Peer.path(root, args.peer))
        train, watched = peer.folds()
        drawer = SAMPLERS[args.sampler](train)
        drawn = Simulated(CORPORA[args.corpus](study, train), drawer)
        simulated, held = (
            drawn.held(args.held, torch.Generator().manual_seed(args.seed)) if args.held else (drawn, watched)
        )
        graders = Study({BASES[0]: val, BASES[1]: peer.rebased(val)}, test)
        steps = math.ceil(args.passes * simulated.count() / BATCH)
        epoch = math.ceil(train.count() / BATCH)
        arms = "+".join(args.arms)
        stem = f"{STUDY}-{args.sampler}-{args.corpus}-h{args.held}-p{args.passes:g}-e{args.epochs}-s{args.seed}"
        stem = f"{stem}-{args.peer}-{arms}"
        profile = {"real": asdict(Profile.of(watched)), args.sampler: asdict(Profile.simulated(watched, drawer, 0))}
        result: dict[str, object] = {
            "aligned": {VAL: val_aligned, TEST: test_aligned},
            "spans": simulated.count(),
            "profile": profile,
        }
        for showing in (SHOWINGS[name]() for name in args.showings):
            trainer = Trainer(showing, watched, args.seed)
            base = watched.loss(Refiner().to(device), showing)
            pretrainer = Trainer(showing, held, args.seed)
            stage = functools.cache(
                lambda pretrainer=pretrainer: pretrainer.picked(
                    Refiner().to(device), simulated, steps, max(1, steps // CHECKS)
                )
            )
            models = {
                "text-only": stage,
                "text": lambda trainer=trainer, stage=stage: trainer.picked(
                    stage()[0], train, args.epochs * epoch, epoch
                ),
                "meg-only": lambda trainer=trainer: trainer.picked(
                    Refiner().to(device), train, args.epochs * epoch, epoch
                ),
            }
            for arm in args.arms:
                model, fit = models[arm]()
                fit["real"] = watched.loss(model, showing)
                key = f"{showing.name}/{arm}"
                torch.save(model.state_dict(), study / f"{args.name}-{stem}-{showing.name}-{arm}.pt")
                result[key] = {"fit": fit, "watched-base": base, **graders.scored(model, showing, args.seed)}
                LOG.info("%s %s", key, json.dumps(result[key])[:2000])
                (out / f"{stem}.json").write_text(json.dumps(result, indent=1))
        LOG.info("wrote %s", out / f"{stem}.json")


if __name__ == "__main__":
    Study.main()
