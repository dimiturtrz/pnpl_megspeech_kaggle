"""Fit the sentence-context word classifier: fixed-shape optimiser steps, best epoch kept by validation.

Every word is read inside its span (`megspeech.data.spans`): a CNN embeds each 1 s window, a transformer refines
the embeddings across the span, and both the isolated-word head and the context head are trained.
"""

import logging
import math
from dataclasses import asdict, dataclass

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn

from core.paths import DataRoot
from megspeech.data.corpus import Corpus
from megspeech.data.split import Split
from megspeech.data.vocabulary import IGNORE
from megspeech.evaluation.report import Epoch, Evaluator
from megspeech.model.scaler import SCALER_SAMPLE, Scaler
from megspeech.model.sentence import SentenceNet
from megspeech.objective.heads import Objective
from megspeech.objective.prototypes import Prototypes
from megspeech.training.config import RunConfig

LOG = logging.getLogger(__name__)


@dataclass
class Step:
    """What one optimiser step mutates, and the word rows of its fixed-shape batch."""

    model: nn.Module
    optimizer: torch.optim.Optimizer
    schedule: torch.optim.lr_scheduler.LRScheduler
    rows: int


@dataclass
class Trainer:
    config: RunConfig
    evaluator: Evaluator
    prototypes: Float[Tensor, "v d"] | None = None

    @classmethod
    def of(cls, corpus: Corpus, config: RunConfig, root: DataRoot) -> "Trainer":
        """A trainer on `corpus`'s device, selecting on its unseen spans; SigLIP reads the text prototypes."""
        siglip = config.context.objective is Objective.SIGLIP
        prototypes = Prototypes.load(Prototypes.path(root)).vectors if siglip else None
        evaluator = Evaluator(corpus.train.y.device, corpus.seen, config.selection.min_novel)
        return cls(config, evaluator, prototypes)

    def model(self, train: Split) -> nn.Module:
        device = self.evaluator.device
        model = SentenceNet(
            Scaler(train.sample(SCALER_SAMPLE)), context=self.config.context, prototypes=self.prototypes
        ).to(device)
        if self.config.optim.compiled and device.type == "cuda":
            model.compile(mode="reduce-overhead")
        return model

    def optimizer(self, model: nn.Module) -> torch.optim.AdamW:
        """AdamW as one fused kernel on the GPU; per-parameter launches were most of a step's device time."""
        optim = self.config.optim
        fused = self.evaluator.device.type == "cuda"
        return torch.optim.AdamW(model.parameters(), lr=optim.lr, weight_decay=optim.weight_decay, fused=fused)

    def step(self, state: Step, train: Split, spans: Int[Tensor, " b"]) -> Float[Tensor, ""]:
        """One optimiser step on the words of `spans`; returns the loss without syncing it to the host."""
        batch = train.batch(spans, state.rows)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            context, word = state.model(self.config.augment(batch.x), batch.onset, batch.slot)
            smoothing, objective, y = (
                self.config.optim.label_smoothing,
                self.config.context.objective,
                batch.target(train.y),
            )
            value = objective.loss(context.flatten(0, 1), y.flatten(), smoothing) + nn.functional.cross_entropy(
                word.flatten(0, 1), y.flatten(), ignore_index=IGNORE, label_smoothing=smoothing
            )
        state.optimizer.zero_grad(set_to_none=True)
        value.backward()
        nn.utils.clip_grad_norm_(state.model.parameters(), self.config.optim.clip)
        state.optimizer.step()
        state.schedule.step()
        return value.detach()

    def fit(
        self, train: Split, validation: dict[str, Split], log_prior: Float[Tensor, " v"]
    ) -> tuple[nn.Module, list[Epoch]]:
        optim = self.config.optim
        torch.manual_seed(optim.seed)
        model = self.model(train)
        optimizer = self.optimizer(model)
        per_epoch = math.ceil(len(train.index) / optim.batch)
        schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, optim.lr, total_steps=optim.epochs * per_epoch)
        state = Step(model, optimizer, schedule, train.capacity(optim.batch))
        history: list[Epoch] = []
        best, best_state = -1.0, model.state_dict()
        for epoch in range(optim.epochs):
            model.train()
            draws = torch.multinomial(train.weight[train.index[:, 0]], per_epoch * optim.batch, replacement=True)
            value = torch.stack([self.step(state, train, spans) for spans in draws.split(optim.batch)]).mean().item()
            if not math.isfinite(value):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}")
            row = self.evaluator.report(model, validation, log_prior, epoch, value)
            history.append(row)
            LOG.info("%s", asdict(row))
            if row.macro.bacc10 > best:
                best = row.macro.bacc10
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        return model, history
