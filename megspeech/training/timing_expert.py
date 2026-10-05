"""A timing expert trained alone: the `TimingHead` fitted to every labelled word from its onsets and span, no MEG."""

import logging
import math
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

    @staticmethod
    def load(root: DataRoot, name: str) -> dict[str, Tensor]:
        """The head weights saved under models/`name`, none for an empty name."""
        return torch.load(root.models() / name / FILE, weights_only=True) if name else {}

    @staticmethod
    def save(head: nn.Module, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(head.state_dict(), directory / FILE)

    def loss(self, model: TimingOnly, train: Split, spans: Int[Tensor, " s"]) -> Float[Tensor, ""]:
        index = train.index[spans]
        mask = index != PAD
        labels = torch.where(mask, train.y[index.clamp(min=0)], IGNORE)
        logits = model.timing(train.onset[spans], mask)
        return nn.functional.cross_entropy(
            logits.flatten(0, 1).float(), labels.flatten(), label_smoothing=self.label_smoothing
        )

    def fit(
        self, train: Split, validation: dict[str, Split], evaluator: Evaluator, log_prior: Float[Tensor, " v"]
    ) -> tuple[TimingOnly, list[Epoch]]:
        """The model at its best epoch on `validation`, and every epoch's report."""
        torch.manual_seed(self.seed)
        model = TimingOnly(VOCAB_SIZE).to(train.y.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        per_epoch = math.ceil(len(train.index) / self.batch)
        schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, self.lr, total_steps=self.epochs * per_epoch)
        history: list[Epoch] = []
        best = {k: v.clone() for k, v in model.state_dict().items()}
        for epoch in range(self.epochs):
            model.train()
            total = torch.zeros((), device=train.y.device)
            for spans in torch.randperm(len(train.index), device=train.y.device).split(self.batch):
                value = self.loss(model, train, spans)
                optimizer.zero_grad(set_to_none=True)
                value.backward()
                optimizer.step()
                schedule.step()
                total += value.detach()
            row = evaluator.report(model, validation, log_prior, epoch, total.item() / per_epoch)
            LOG.info("epoch %d loss %.4f macro %.4f", epoch, row.loss, row.macro.bacc10)
            if not history or row.macro.bacc10 > max(h.macro.bacc10 for h in history):
                best = {k: v.clone() for k, v in model.state_dict().items()}
            history.append(row)
        model.load_state_dict(best)
        return model, history
