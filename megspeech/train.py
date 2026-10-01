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
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Protocol

import numpy as np
import pandas as pd
import torch
from jaxtyping import Float, Int
from pnpl.competition.holdout import LibriBrainCompetitionHoldout
from torch import Tensor, nn

from core.paths import DataRoot
from megspeech.epochs import Epocher, Epochs, Window
from megspeech.model import SentenceNet
from megspeech.runs import Partition, Run
from megspeech.scaler import Scaler
from megspeech.score import BalancedTopK
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
    seed: int = 0


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
    """One validation domain: both heads, the context head on holdout-like spans, and its top-1."""

    context: Score
    word: Score
    dense: Score
    bacc1: float


@dataclass(frozen=True)
class Epoch:
    """One validation pass over every domain, and the context head's macro over domains that selects."""

    epoch: int
    loss: float
    domains: dict[str, Report]
    macro: Score


@dataclass(frozen=True)
class Split:
    """Word windows on the device, with the spans that group them."""

    x: Float[Tensor, "n c t"]
    y: Int[Tensor, " n"]
    index: Int[Tensor, "g w"]
    onset: Float[Tensor, "g w"]

    @classmethod
    def from_arrays(
        cls, x: Float[np.ndarray, "n c t"], y: Int[np.ndarray, " n"], spans: Spans, device: torch.device
    ) -> "Split":
        return cls.on_device(torch.from_numpy(x).to(device), y, spans)

    @classmethod
    def on_device(cls, x: Float[Tensor, "n c t"], y: Int[np.ndarray, " n"], spans: Spans) -> "Split":
        device = x.device
        return cls(
            x,
            torch.from_numpy(y).to(device),
            torch.from_numpy(spans.index).to(device),
            torch.from_numpy(spans.onset).float().to(device),
        )

    @classmethod
    def gather(cls, epochs: Iterable[Epochs], device: torch.device) -> "Split":
        """Runs streamed to the device one at a time, so host memory never holds the whole split."""
        xs: list[Tensor] = []
        ys: list[np.ndarray] = []
        onsets: list[np.ndarray] = []
        for run in epochs:
            xs.append(torch.from_numpy(run.x).to(device))
            ys.append(run.y)
            onsets.append(run.onset)
        return cls.on_device(torch.cat(xs), np.concatenate(ys), Spans.by_gap(np.concatenate(onsets)))

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
        counts = torch.bincount(self.y, minlength=WORDS).float() + 1
        return (counts / counts.sum()).log()

    def batch(self, spans: Int[Tensor, " b"]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        index = self.index[spans]
        mask = index != PAD
        words = index[mask]
        return self.x[words], self.onset[spans], mask, words

    def dense(self) -> Int[Tensor, " d"]:
        """Words in spans at least `DENSE` long — the holdout-like subset."""
        long = self.index[(self.index != PAD).sum(dim=1) >= DENSE]
        return long[long != PAD]


@dataclass
class Trainer:
    config: TrainConfig
    device: torch.device

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
        metric = BalancedTopK()
        truth = labels.cpu().numpy()
        by_tau = {tau: metric((logits - tau * log_prior).cpu().numpy(), truth) for tau in ADJUSTMENTS}
        return Score.best(by_tau, len(truth))

    def assess(self, model: nn.Module, split: Split, log_prior: Float[Tensor, " v"]) -> Report:
        logits = self.logits(model, split)
        dense = split.dense()
        return Report(
            self.evaluate(logits[Head.CONTEXT], split.y, log_prior),
            self.evaluate(logits[Head.WORD], split.y, log_prior),
            self.evaluate(logits[Head.CONTEXT][dense], split.y[dense], log_prior),
            BalancedTopK(1)(logits[Head.CONTEXT].cpu().numpy(), split.y.cpu().numpy()),
        )

    def report(
        self, model: nn.Module, domains: dict[str, Split], log_prior: Float[Tensor, " v"], epoch: int, loss: float
    ) -> Epoch:
        """Every domain scored alone; read selection off the macro so the biggest domain cannot drown the rest."""
        reports = {name: self.assess(model, split, log_prior) for name, split in domains.items()}
        by_tau = {tau: float(np.mean([r.context.by_tau[tau] for r in reports.values()])) for tau in ADJUSTMENTS}
        return Epoch(epoch, loss, reports, Score.best(by_tau, sum(r.context.n for r in reports.values())))

    def fit(
        self, train: Split, validation: dict[str, Split], log_prior: Float[Tensor, " v"]
    ) -> tuple[nn.Module, list[Epoch]]:
        torch.manual_seed(self.config.seed)
        model = SentenceNet(Scaler(train.x)).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        steps = self.config.epochs * math.ceil(len(train.index) / self.config.batch)
        schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, self.config.lr, total_steps=steps)
        loss = nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)
        history: list[Epoch] = []
        best, best_state = -1.0, model.state_dict()
        for epoch in range(self.config.epochs):
            model.train()
            for spans in torch.randperm(len(train.index), device=self.device).split(self.config.batch):
                x, onset, mask, words = train.batch(spans)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    context, word = model(x, onset, mask)
                    value = loss(context[mask], train.y[words]) + loss(word[mask], train.y[words])
                optimizer.zero_grad(set_to_none=True)
                value.backward()
                optimizer.step()
                schedule.step()
            row = self.report(model, validation, log_prior, epoch, value.item())
            history.append(row)
            LOG.info("%s", asdict(row))
            if row.macro.bacc10 > best:
                best = row.macro.bacc10
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        return model, history


def main() -> None:
    """Train, write metrics + checkpoint + holdout submission under models/pnpl_libribrain/<name>."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    config = TrainConfig(epochs=args.epochs)
    device = torch.device("cuda")
    root = DataRoot.from_config()
    epocher = Epocher(root.processed() / "epochs")
    runs = Run.discover(root.raw())
    train = Split.gather((epocher.cached(r) for r in runs if r.partition is Partition.TRAIN), device)
    held = [r for r in runs if r.partition is Partition.VALIDATION]
    validation = {
        task: Split.gather((epocher.cached(r) for r in held if r.key[0] == task), device)
        for task in sorted({r.key[0] for r in held})
    }
    LOG.info("train %d words / %d spans", len(train.y), len(train.index))
    LOG.info("validation %s", {task: len(split.y) for task, split in validation.items()})
    log_prior = train.log_prior()
    trainer = Trainer(config, device)
    model, history = trainer.fit(train, validation, log_prior)
    out = root.models() / args.name
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    best = max(history, key=lambda row: row.macro.bacc10)
    summary = {"config": asdict(config), "train": len(train.y), "best": asdict(best)}
    (out / "metrics.json").write_text(json.dumps({**summary, "history": [asdict(row) for row in history]}, indent=1))
    LOG.info("best %s", best)
    holdout = LibriBrainCompetitionHoldout("deep", data_path=root.raw(), download=False)
    logits = trainer.logits(model, Split.from_holdout(holdout, device))[Head.CONTEXT]
    primary = torch.softmax(logits - best.macro.tau * log_prior, dim=-1).cpu().numpy()
    secondary = np.tile(FrequencyPrior(Counter()).probabilities(list(SECONDARY.words)), (len(primary), 1))
    Submission(holdout).write(out / "submission.csv", primary, secondary)


if __name__ == "__main__":
    main()
