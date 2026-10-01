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


@dataclass(frozen=True)
class Epoch:
    """One validation pass: both heads, the context head on holdout-like spans, and its top-1."""

    epoch: int
    loss: float
    context: Score
    word: Score
    dense: Score
    bacc1: float


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
        return cls(
            torch.from_numpy(x).to(device),
            torch.from_numpy(y).to(device),
            torch.from_numpy(spans.index).to(device),
            torch.from_numpy(spans.onset).float().to(device),
        )

    @classmethod
    def gather(cls, epochs: list[Epochs], device: torch.device) -> "Split":
        onset = np.concatenate([e.onset for e in epochs])
        x, y = np.concatenate([e.x for e in epochs]), np.concatenate([e.y for e in epochs])
        return cls.from_arrays(x, y, Spans.by_gap(onset), device)

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
        tau = max(by_tau, key=by_tau.__getitem__)
        return Score(tau, by_tau[tau], by_tau, len(truth))

    def report(self, model: nn.Module, split: Split, log_prior: Float[Tensor, " v"], epoch: int, loss: float) -> Epoch:
        logits = self.logits(model, split)
        dense = split.dense()
        return Epoch(
            epoch,
            loss,
            self.evaluate(logits[Head.CONTEXT], split.y, log_prior),
            self.evaluate(logits[Head.WORD], split.y, log_prior),
            self.evaluate(logits[Head.CONTEXT][dense], split.y[dense], log_prior),
            BalancedTopK(1)(logits[Head.CONTEXT].cpu().numpy(), split.y.cpu().numpy()),
        )

    def fit(self, train: Split, validation: Split, log_prior: Float[Tensor, " v"]) -> tuple[nn.Module, list[Epoch]]:
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
            if row.context.bacc10 > best:
                best = row.context.bacc10
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
    train, validation = (
        Split.gather([epocher.cached(r) for r in runs if r.partition is partition], device)
        for partition in (Partition.TRAIN, Partition.VALIDATION)
    )
    LOG.info(
        "train %d words / %d spans, validation %d / %d",
        len(train.y),
        len(train.index),
        len(validation.y),
        len(validation.index),
    )
    log_prior = train.log_prior()
    trainer = Trainer(config, device)
    model, history = trainer.fit(train, validation, log_prior)
    out = root.models() / args.name
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    best = max(history, key=lambda row: row.context.bacc10)
    summary = {"config": asdict(config), "train": len(train.y), "validation": len(validation.y), "best": asdict(best)}
    (out / "metrics.json").write_text(json.dumps({**summary, "history": [asdict(row) for row in history]}, indent=1))
    LOG.info("best %s", best)
    holdout = LibriBrainCompetitionHoldout("deep", data_path=root.raw(), download=False)
    logits = trainer.logits(model, Split.from_holdout(holdout, device))[Head.CONTEXT]
    primary = torch.softmax(logits - best.context.tau * log_prior, dim=-1).cpu().numpy()
    secondary = np.tile(FrequencyPrior(Counter()).probabilities(list(SECONDARY.words)), (len(primary), 1))
    Submission(holdout).write(out / "submission.csv", primary, secondary)


if __name__ == "__main__":
    main()
