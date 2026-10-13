"""A timing expert trained alone: the `TimingHead` fitted to every labelled word from its onsets and span, no MEG."""

import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn

from core.paths import DataRoot
from megspeech.data.batching.split import Split
from megspeech.data.labels.vocabulary import IGNORE, VOCAB_SIZE
from megspeech.data.spans.spans import PAD
from megspeech.evaluation.scoring.evaluator import Evaluator
from megspeech.evaluation.scoring.score import Epoch
from megspeech.model.context.timing_only import TimingOnly
from megspeech.training.timing_corpus import Timed, TimingCorpus

LOG = logging.getLogger(__name__)

FILE = "timing.pt"


@dataclass(frozen=True)
class TimingExpert:
    """Cross-entropy on the context labels of every span the model draws, AdamW on a cosine schedule."""

    epochs: int = 20
    batch: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.05
    label_smoothing: float = 0.1
    seed: int = 0
    layers: int = 0
    """Transformer layers of a span-wide `TimingSequence`; 0 trains the local `TimingHead`."""
    pretrain: str = ""
    """A corpus under processed/ (`TimingCorpus.read`) first learned for `pretrain_epochs`; empty is none."""
    pretrain_epochs: int = 3

    @staticmethod
    def load(root: DataRoot, name: str) -> dict[str, Tensor]:
        """The head weights saved under models/`name`, none for an empty name."""
        return torch.load(root.models() / name / FILE, weights_only=True) if name else {}

    @staticmethod
    def save(head: nn.Module, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(head.state_dict(), directory / FILE)

    def loss(self, model: TimingOnly, train: Timed, spans: Int[Tensor, " s"]) -> Float[Tensor, ""]:
        index = train.index[spans]
        mask = index != PAD
        labels = torch.where(mask, train.y[index.clamp(min=0)], IGNORE)
        logits = model.timing(train.onset[spans], mask)
        return nn.functional.cross_entropy(
            logits.flatten(0, 1).float(), labels.flatten(), label_smoothing=self.label_smoothing
        )

    def run_epoch(
        self,
        model: TimingOnly,
        train: Timed,
        optimizer: torch.optim.Optimizer,
        schedule: torch.optim.lr_scheduler.LRScheduler,
    ) -> float:
        """One pass over every span of `train`, its mean batch loss."""
        model.train()
        device = train.y.device
        total = torch.zeros((), device=device)
        for spans in torch.randperm(len(train.index), device=device).split(self.batch):
            value = self.loss(model, train, spans)
            optimizer.zero_grad(set_to_none=True)
            value.backward()
            optimizer.step()
            schedule.step()
            total += value.detach()
        return total.item() / math.ceil(len(train.index) / self.batch)

    def optimiser(
        self, model: TimingOnly, train: Timed, epochs: int
    ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
        """AdamW and a one-cycle schedule over `epochs` passes of `train`."""
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        per_epoch = math.ceil(len(train.index) / self.batch)
        return optimizer, torch.optim.lr_scheduler.OneCycleLR(optimizer, self.lr, total_steps=epochs * per_epoch)

    def pretrained(self, model: TimingOnly, corpus: TimingCorpus) -> list[float]:
        """Train `model` on `corpus` for `pretrain_epochs`, no selection; every epoch's mean loss."""
        optimizer, schedule = self.optimiser(model, corpus, self.pretrain_epochs)
        losses = []
        for epoch in range(self.pretrain_epochs):
            started = time.monotonic()
            losses.append(self.run_epoch(model, corpus, optimizer, schedule))
            LOG.info("pretrain epoch %d loss %.4f seconds %.1f", epoch, losses[-1], time.monotonic() - started)
        return losses

    def fit(
        self,
        train: Split,
        validation: dict[str, Split],
        evaluator: Evaluator,
        log_prior: Float[Tensor, " v"],
        corpus: TimingCorpus | None = None,
    ) -> tuple[TimingOnly, list[Epoch]]:
        """The model at its best epoch on `validation`, and every epoch's report; trained first on `corpus` if given."""
        torch.manual_seed(self.seed)
        model = TimingOnly(VOCAB_SIZE, self.layers).to(train.y.device)
        if corpus is not None:
            self.pretrained(model, corpus.to(train.y.device))
            zero = evaluator.report(model, validation, log_prior, -1, float("nan"))
            LOG.info("zero-shot macro %.4f", zero.macro.bacc10)
        optimizer, schedule = self.optimiser(model, train, self.epochs)
        history: list[Epoch] = []
        best = {k: v.clone() for k, v in model.state_dict().items()}
        for epoch in range(self.epochs):
            loss = self.run_epoch(model, train, optimizer, schedule)
            row = evaluator.report(model, validation, log_prior, epoch, loss)
            LOG.info("epoch %d loss %.4f macro %.4f", epoch, row.loss, row.macro.bacc10)
            if not history or row.macro.bacc10 > max(h.macro.bacc10 for h in history):
                best = {k: v.clone() for k, v in model.state_dict().items()}
            history.append(row)
        model.load_state_dict(best)
        return model, history
