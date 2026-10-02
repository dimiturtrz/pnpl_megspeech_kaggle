"""Train the sentence-context word classifier on the official train split, select on validation, predict the holdout.

Every word is read inside its span (`megspeech.spans`): a CNN embeds each 1 s window, a transformer refines
the embeddings across the span, and both the isolated-word head and the context head are trained and scored.

Balanced accuracy rewards rare words as much as frequent ones, so scores are read both raw and logit-adjusted
(Menon et al. 2021): subtracting the log training prior turns p(word | MEG) into a score proportional to
p(MEG | word), the Bayes rule for a balanced metric.

    uv run python -m megspeech.train --name context-v0
"""

import argparse
import json
import logging
import math
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
import torch
from jaxtyping import Float, Int
from pnpl.competition.holdout import LibriBrainCompetitionHoldout
from torch import Tensor, nn

from core.paths import DataRoot
from megspeech.augment import Augment
from megspeech.epochs import Epocher, Epochs, Window
from megspeech.heads import Objective
from megspeech.model import SentenceNet
from megspeech.onsets import CONTEXT, Clock, ContextConfig
from megspeech.prototypes import Prototypes
from megspeech.ruler import Coverage, Graded, Interval, Ruler
from megspeech.runs import Partition, Run
from megspeech.scaler import Scaler
from megspeech.spans import PAD, Spans
from megspeech.submission import FrequencyPrior, Submission
from megspeech.vocabulary import SECONDARY
from megspeech.wordnet import WORDS

LOG = logging.getLogger(__name__)

EVAL_SPANS = 256
ADJUSTMENTS = (0.0, 0.5, 1.0)
"""Logit-adjustment strengths tried on validation: none, half, and the full Bayes correction."""

DENSE = 4
"""Validation spans this long or longer look like holdout sentences (94% of which carry 4-9 words)."""


RECITED = 3
"""Spans this long recur only when a sentence is replayed; 2-word pairs of 50 words (2500) recur by chance."""

READ = frozenset({"TIMIT", "MOCHATIMIT"})
"""Tasks of read sentences, the holdout's domain; the rest is narrative (Sherlock, TheMoth)."""


class Head(StrEnum):
    CONTEXT = "context"
    WORD = "word"


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 40
    batch: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.05
    label_smoothing: float = 0.1
    clip: float = math.inf
    augment: Augment = Augment()
    seed: int = 0
    read_weight: float = 1.0
    """Sampling weight of read-sentence runs (TIMIT, MOCHA-TIMIT) — the holdout's domain — against narrative ones."""
    repeat_power: float = 0.0
    """A span heard `k` times in training is drawn with weight `k ** -repeat_power`; 1 draws each sentence once."""
    min_novel: int = 100
    """Unseen-span words a domain needs to count toward selection: two per class of the 50 on average."""
    context: ContextConfig = CONTEXT
    tasks: tuple[str, ...] = ()
    """Training restricted to these tasks' runs; empty trains on every task."""
    compiled: bool = False
    """Compile the model's forward on the GPU (no gain measured yet: shapes change every batch)."""

    def trains_on(self, run: Run) -> bool:
        return run.partition is Partition.TRAIN and (not self.tasks or run.key[0] in self.tasks)


class Windows(Protocol):
    """What a split needs of the holdout: its row metadata and its MEG windows in that order."""

    def metadata_frame(self) -> pd.DataFrame: ...

    def iter_windows(self, _batch_size: int, /) -> Iterable[tuple[np.ndarray, object]]: ...


@dataclass(frozen=True)
class Score:
    tau: float
    bacc10: float
    by_tau: dict[float, float]
    n: int

    @classmethod
    def best(cls, by_tau: dict[float, float], n: int) -> "Score":
        tau = max(by_tau, key=by_tau.__getitem__)
        return cls(tau, by_tau[tau], by_tau, n)


@dataclass(frozen=True)
class Report:
    """One validation domain: both heads, the context head on holdout-like and on unseen spans, and its top-1."""

    context: Score
    word: Score
    dense: Score
    novel: Score
    bacc1: float
    coverage: Coverage


@dataclass(frozen=True)
class Epoch:
    """One validation pass over every domain, the macro over domains' unseen spans that selects, and its interval."""

    epoch: int
    loss: float
    domains: dict[str, Report]
    macro: Score
    interval: Interval


@dataclass(frozen=True)
class Split:
    """Word windows on the device, with the spans that group them and each word's sampling weight."""

    x: Float[Tensor, "n c t"]
    y: Int[Tensor, " n"]
    index: Int[Tensor, "g w"]
    onset: Float[Tensor, "g w"]
    weight: Float[Tensor, " n"]

    @classmethod
    def from_arrays(
        cls, x: Float[np.ndarray, "n c t"], y: Int[np.ndarray, " n"], spans: Spans, device: torch.device
    ) -> "Split":
        return cls.on_device(torch.from_numpy(x).to(device), y, spans, np.ones(len(y)))

    @classmethod
    def on_device(
        cls, x: Float[Tensor, "n c t"], y: Int[np.ndarray, " n"], spans: Spans, weight: Float[np.ndarray, " n"]
    ) -> "Split":
        device = x.device
        return cls(
            x,
            torch.from_numpy(y).to(device),
            torch.from_numpy(spans.index).to(device),
            torch.from_numpy(spans.onset).float().to(device),
            torch.from_numpy(weight).float().to(device),
        )

    @classmethod
    def gather(cls, runs: Iterable[tuple[Epochs, float]], device: torch.device) -> "Split":
        """Runs, each with its sampling weight, streamed to the device one at a time.

        Host memory never holds the whole split. Onsets restart with every run, so no span crosses two.
        """
        xs: list[Tensor] = []
        ys: list[np.ndarray] = []
        onsets: list[np.ndarray] = []
        weights: list[np.ndarray] = []
        for run, weight in runs:
            xs.append(torch.from_numpy(run.x).to(device))
            ys.append(run.y)
            onsets.append(run.onset)
            weights.append(np.full(len(run.y), weight))
        spans = Spans.by_gap(np.concatenate(onsets))
        x = torch.cat(xs)
        xs.clear()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return cls.on_device(x, np.concatenate(ys), spans, np.concatenate(weights))

    @classmethod
    def validation(cls, epocher: Epocher, runs: list[Run], device: torch.device) -> dict[str, "Split"]:
        """The official validation runs, one split per task so each domain is scored alone."""
        held = [r for r in runs if r.partition is Partition.VALIDATION]
        return {
            task: cls.gather(((epocher.cached(r), 1.0) for r in held if r.key[0] == task), device)
            for task in sorted({r.key[0] for r in held})
        }

    @classmethod
    def from_holdout(cls, holdout: Windows, device: torch.device) -> "Split":
        """Every holdout window, grouped by sentence; isolated word rows are spans of one."""
        frame = holdout.metadata_frame()
        sentence = frame["source"] == "sentence"
        group = np.where(sentence, frame["epoch"], -1 - frame["index"])
        keys = frame.assign(group=group)[["subject", "source", "group"]]
        window = Window()
        x = np.concatenate([window(batch) for batch, _ in holdout.iter_windows(EVAL_SPANS)])
        spans = Spans.by_key(keys, frame["onset_s"].to_numpy())
        return cls.from_arrays(x, np.zeros(len(x), dtype=np.int64), spans, device)

    def log_prior(self) -> Float[Tensor, " v"]:
        """Label prior as training samples it, so logit adjustment undoes the same distribution."""
        return self.prior(self.y, self.weight)

    @staticmethod
    def prior(y: Int[Tensor, " n"], weight: Float[Tensor, " n"]) -> Float[Tensor, " v"]:
        counts = torch.bincount(y, weight, minlength=WORDS).float() + 1
        return (counts / counts.sum()).log()

    def batch(self, spans: Int[Tensor, " b"]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        index = self.index[spans]
        mask = index != PAD
        words = index[mask]
        return self.x[words], self.onset[spans], mask, words

    @staticmethod
    def seen(labelled: Iterable[tuple[Int[np.ndarray, " n"], Float[np.ndarray, " n"]]]) -> frozenset[tuple[int, ...]]:
        """Every span's label sequence across runs of (labels, onsets) — the word orders training has heard."""
        sequences: set[tuple[int, ...]] = set()
        for y, onset in labelled:
            index = Spans.by_gap(onset).index
            sequences.update(tuple(y[row[row != PAD]].tolist()) for row in index)
        return frozenset(sequences)

    def novel(self, seen: frozenset[tuple[int, ...]]) -> Int[Tensor, " k"]:
        """Words in multi-word spans whose label sequence training never heard.

        The holdout's sentences are new stimuli, so a span recited from training measures memorised word order.
        """
        index = self.index.cpu().numpy()
        labels = self.y.cpu().numpy()
        rows = [row[row != PAD] for row in index]
        keep = [row for row in rows if len(row) > 1 and tuple(labels[row].tolist()) not in seen]
        return torch.from_numpy(np.concatenate([np.empty(0, np.int64), *keep])).to(self.y.device)

    def recurring(self, power: float) -> "Split":
        """Each recited span's words reweighted by how often training repeats its label sequence, to the `-power`."""
        index = self.index.cpu().numpy()
        labels = self.y.cpu().numpy()
        rows = [row[row != PAD] for row in index]
        sequences = [tuple(labels[row].tolist()) for row in rows]
        counts = Counter(sequences)
        scale = np.ones(len(labels))
        for row, sequence in zip(rows, sequences, strict=True):
            scale[row] = counts[sequence] ** -power if len(row) >= RECITED else 1.0
        return replace(self, weight=self.weight * torch.from_numpy(scale).float().to(self.weight.device))

    def dense(self) -> Int[Tensor, " d"]:
        """Words in spans at least `DENSE` long — the holdout-like subset."""
        long = self.index[(self.index != PAD).sum(dim=1) >= DENSE]
        return long[long != PAD]


@dataclass
class Step:
    """What one optimiser step mutates."""

    model: nn.Module
    optimizer: torch.optim.Optimizer
    schedule: torch.optim.lr_scheduler.LRScheduler


@dataclass
class Trainer:
    config: TrainConfig
    device: torch.device
    seen: frozenset[tuple[int, ...]] = frozenset()
    prototypes: Float[Tensor, "v d"] | None = None
    ruler: Ruler = Ruler()

    def logits(self, model: nn.Module, split: Split) -> dict[Head, Float[Tensor, "n v"]]:
        """Both heads' logits for every word of `split`, in the split's word order."""
        model.eval()
        out = {head: torch.zeros(len(split.x), WORDS, device=self.device) for head in Head}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for spans in torch.arange(len(split.index), device=self.device).split(EVAL_SPANS):
                x, onset, mask, words = split.batch(spans)
                context, word = model(x, onset, mask)
                out[Head.CONTEXT][words] = context[mask].float()
                out[Head.WORD][words] = word[mask].float()
        return out

    def evaluate(
        self, logits: Float[Tensor, "n v"], labels: Int[Tensor, " n"], log_prior: Float[Tensor, " v"]
    ) -> Score:
        by_tau = {
            tau: self.ruler.balanced(self.ruler.hits(logits - tau * log_prior, labels), labels) for tau in ADJUSTMENTS
        }
        return Score.best(by_tau, len(labels))

    def assess(self, model: nn.Module, split: Split, log_prior: Float[Tensor, " v"]) -> Report:
        logits = self.logits(model, split)
        dense = split.dense()
        novel = split.novel(self.seen)
        top1 = Ruler(k=1)
        return Report(
            self.evaluate(logits[Head.CONTEXT], split.y, log_prior),
            self.evaluate(logits[Head.WORD], split.y, log_prior),
            self.evaluate(logits[Head.CONTEXT][dense], split.y[dense], log_prior),
            self.evaluate(logits[Head.CONTEXT][novel], split.y[novel], log_prior),
            top1.balanced(top1.hits(logits[Head.CONTEXT], split.y), split.y),
            Coverage.of(split.y[novel]),
        )

    def graded(self, model: nn.Module, domains: dict[str, Split], log_prior: Float[Tensor, " v"], tau: float) -> Graded:
        """The context head's per-word hits on the unseen spans of every domain that counts toward selection."""
        novel = {name: split.novel(self.seen) for name, split in domains.items()}
        counted = {name: split for name, split in domains.items() if len(novel[name]) >= self.config.min_novel}
        return Graded(
            {
                name: (
                    self.ruler.hits(
                        self.logits(model, split)[Head.CONTEXT][novel[name]] - tau * log_prior, split.y[novel[name]]
                    ),
                    split.y[novel[name]],
                )
                for name, split in counted.items()
            }
        )

    def report(
        self, model: nn.Module, domains: dict[str, Split], log_prior: Float[Tensor, " v"], epoch: int, loss: float
    ) -> Epoch:
        """Every domain scored alone; selection reads the macro over unseen spans of domains with enough of them."""
        reports = {name: self.assess(model, split, log_prior) for name, split in domains.items()}
        counted = [r.novel for r in reports.values() if r.novel.n >= self.config.min_novel]
        by_tau = {tau: float(np.mean([score.by_tau[tau] for score in counted])) for tau in ADJUSTMENTS}
        macro = Score.best(by_tau, sum(score.n for score in counted))
        interval = self.ruler.interval(self.ruler.macro(self.graded(model, domains, log_prior, macro.tau)))
        return Epoch(epoch, loss, reports, macro, interval)

    def model(self, train: Split) -> nn.Module:
        model = SentenceNet(Scaler(train.x), context=self.config.context, prototypes=self.prototypes).to(self.device)
        if self.config.compiled and self.device.type == "cuda":
            model.compile(dynamic=True)
        return model

    def optimizer(self, model: nn.Module) -> torch.optim.AdamW:
        """AdamW as one fused kernel on the GPU; per-parameter launches were most of a step's device time."""
        fused = self.device.type == "cuda"
        return torch.optim.AdamW(
            model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay, fused=fused
        )

    def step(self, state: "Step", train: Split, spans: Int[Tensor, " b"]) -> Float[Tensor, ""]:
        """One optimiser step on the words of `spans`; returns the loss without syncing it to the host."""
        x, onset, mask, words = train.batch(spans)
        x = self.config.augment(x)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            context, word = state.model(x, onset, mask)
            smoothing, objective, y = self.config.label_smoothing, self.config.context.objective, train.y[words]
            value = objective.loss(context[mask], y, smoothing) + nn.functional.cross_entropy(
                word[mask], y, label_smoothing=smoothing
            )
        state.optimizer.zero_grad(set_to_none=True)
        value.backward()
        nn.utils.clip_grad_norm_(state.model.parameters(), self.config.clip)
        state.optimizer.step()
        state.schedule.step()
        return value.detach()

    def fit(
        self, train: Split, validation: dict[str, Split], log_prior: Float[Tensor, " v"]
    ) -> tuple[nn.Module, list[Epoch]]:
        torch.manual_seed(self.config.seed)
        model = self.model(train)
        optimizer = self.optimizer(model)
        steps = self.config.epochs * math.ceil(len(train.index) / self.config.batch)
        schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, self.config.lr, total_steps=steps)
        state = Step(model, optimizer, schedule)
        history: list[Epoch] = []
        best, best_state = -1.0, model.state_dict()
        for epoch in range(self.config.epochs):
            model.train()
            draws = torch.multinomial(train.weight[train.index[:, 0]], len(train.index), replacement=True)
            value = (
                torch.stack([self.step(state, train, spans) for spans in draws.split(self.config.batch)]).mean().item()
            )
            if not math.isfinite(value):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}")
            row = self.report(model, validation, log_prior, epoch, value)
            history.append(row)
            LOG.info("%s", asdict(row))
            if row.macro.bacc10 > best:
                best = row.macro.bacc10
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        return model, history

    def submit(self, model: nn.Module, root: DataRoot, log_prior: Float[Tensor, " v"], tau: float, path: Path) -> Path:
        """Context-head probabilities for every holdout row, logit-adjusted by `tau`, written as a submission."""
        holdout = LibriBrainCompetitionHoldout("deep", data_path=root.raw(), download=False)
        logits = self.logits(model, Split.from_holdout(holdout, self.device))[Head.CONTEXT]
        primary = torch.softmax(logits - tau * log_prior, dim=-1).cpu().numpy()
        secondary = np.tile(FrequencyPrior(Counter()).probabilities(list(SECONDARY.words)), (len(primary), 1))
        return Submission(holdout).write(path, primary, secondary)


@dataclass(frozen=True)
class Corpus:
    """The on-device train set, the validation domains, and every word sequence training has seen."""

    train: Split
    validation: dict[str, Split]
    seen: frozenset[tuple[int, ...]]

    @classmethod
    def load(cls, config: TrainConfig, root: DataRoot, device: torch.device) -> "Corpus":
        epocher = Epocher(root.processed() / "epochs")
        runs = Run.discover(root.raw())
        weighted = (
            (epocher.cached(r), config.read_weight if r.key[0] in READ else 1.0) for r in runs if config.trains_on(r)
        )
        train = Split.gather(weighted, device).recurring(config.repeat_power)
        validation = Split.validation(epocher, runs, device)
        seen = Split.seen(epocher.labelled(r) for r in runs if r.partition is Partition.TRAIN)
        LOG.info("train %d words / %d spans", len(train.y), len(train.index))
        LOG.info("validation %s", {task: len(split.y) for task, split in validation.items()})
        return cls(train, validation, seen)

    def trainer(self, config: TrainConfig, root: DataRoot) -> "Trainer":
        siglip = config.context.objective is Objective.SIGLIP
        prototypes = Prototypes.load(Prototypes.path(root)).vectors if siglip else None
        return Trainer(config, self.train.x.device, self.seen, prototypes)


def main() -> None:
    """Train, write metrics + checkpoint + holdout submission under models/pnpl_libribrain/<name>."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--read-weight", type=float, default=TrainConfig.read_weight)
    parser.add_argument("--repeat-power", type=float, default=TrainConfig.repeat_power)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--dropout", type=float, default=Augment.dropout, help="sensor dropout p")
    parser.add_argument("--gain", type=float, default=Augment.gain, help="log-normal sensor gain sigma")
    parser.add_argument("--clip", type=float, default=TrainConfig.clip, help="max grad norm")
    parser.add_argument("--width", type=int, default=CONTEXT.width)
    parser.add_argument("--heads", type=int, default=CONTEXT.heads)
    parser.add_argument("--layers", type=int, default=CONTEXT.layers)
    parser.add_argument("--clock", type=Clock, default=CONTEXT.clock, choices=list(Clock))
    parser.add_argument("--objective", type=Objective, default=CONTEXT.objective, choices=list(Objective))
    parser.add_argument("--tasks", nargs="*", default=list(TrainConfig.tasks))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    context = replace(
        CONTEXT, width=args.width, heads=args.heads, layers=args.layers, clock=args.clock, objective=args.objective
    )
    config = TrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        clip=args.clip,
        augment=Augment(args.dropout, args.gain),
        read_weight=args.read_weight,
        repeat_power=args.repeat_power,
        seed=args.seed,
        context=context,
        tasks=tuple(args.tasks),
    )
    root = DataRoot.from_config()
    corpus = Corpus.load(config, root, torch.device("cuda"))
    train, validation, log_prior = corpus.train, corpus.validation, corpus.train.log_prior()
    trainer = corpus.trainer(config, root)
    model, history = trainer.fit(train, validation, log_prior)
    out = root.models() / args.name
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    context.save(out)
    best = max(history, key=lambda row: row.macro.bacc10)
    summary = {"config": asdict(config), "train": len(train.y), "best": asdict(best)}
    (out / "metrics.json").write_text(json.dumps({**summary, "history": [asdict(row) for row in history]}, indent=1))
    LOG.info("best %s", best)
    trainer.graded(model, validation, log_prior, best.macro.tau).save(out / "hits.pt")
    trainer.submit(model, root, log_prior, best.macro.tau, out / "submission.csv")


if __name__ == "__main__":
    main()
