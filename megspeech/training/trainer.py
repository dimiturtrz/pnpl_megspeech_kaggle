"""Fit the sentence-context word classifier: fixed-shape optimiser steps, best epoch kept by validation.

Every word is read inside its span (`megspeech.data.spans.spans`): a CNN embeds each 1 s window, a transformer refines
the embeddings across the span, and both the isolated-word head and the context head are trained.
"""

import logging
import math
from dataclasses import asdict, dataclass, field

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn

from core.paths import DataRoot
from megspeech.data.batching.split import Split
from megspeech.data.corpus import Corpus
from megspeech.data.labels.vocabulary import IGNORE, VOCAB_SIZE
from megspeech.evaluation.report import Epoch, Evaluator
from megspeech.model.megxl import Trunk
from megspeech.model.scaler import SCALER_SAMPLE, Scaler
from megspeech.model.sentence import SentenceNet
from megspeech.objective.heads import Objective
from megspeech.objective.prototypes import Prototypes
from megspeech.training.anchor import Anchor
from megspeech.training.auxiliary import Auxiliary, Silent
from megspeech.training.config import RunConfig
from megspeech.training.curriculum import Annealed
from megspeech.training.ema import Ema

LOG = logging.getLogger(__name__)


@dataclass
class Step:
    """What one optimiser step mutates, and the word rows of its fixed-shape batch."""

    model: nn.Module
    optimizer: torch.optim.Optimizer
    schedule: torch.optim.lr_scheduler.LRScheduler
    rows: int
    ema: Ema
    weight: Float[Tensor, " v"] | None = None
    anchor: Anchor = field(default_factory=Anchor)


@dataclass
class Trainer:
    config: RunConfig
    evaluator: Evaluator
    prototypes: Float[Tensor, "v d"] | None = None
    auxiliary: Auxiliary = field(default_factory=Silent)
    initial: dict[str, Tensor] = field(default_factory=dict)

    @classmethod
    def of(cls, corpus: Corpus, config: RunConfig, root: DataRoot) -> "Trainer":
        """A trainer on `corpus`'s device, selecting on its unseen spans; SigLIP reads the text prototypes, a warm
        start its run's weights."""
        siglip = config.context.objective is Objective.SIGLIP
        prototypes = Prototypes.load(Prototypes.path(root)).vectors if siglip else None
        device = corpus.train.y.device
        evaluator = Evaluator(device, corpus.seen, config.selection.min_novel)
        warm = config.optim.warm
        initial = torch.load(root.models() / warm / "model.pt", map_location=device, weights_only=True) if warm else {}
        return cls(config, evaluator, prototypes, Auxiliary.of(config.optim, root, device), initial)

    def model(self, train: Split) -> SentenceNet:
        """A fresh model, or the warm start's weights when there is one."""
        model = SentenceNet(
            Scaler(train.sample(SCALER_SAMPLE)), context=self.config.context, prototypes=self.prototypes
        )
        if self.initial:
            model.load_state_dict(self.initial)
        return self.compiled(model.to(self.evaluator.device))

    def compiled[M: nn.Module](self, model: M) -> M:
        """`model` compiled in place into CUDA graphs when the run compiles and is on the GPU."""
        if self.config.optim.compiled and self.evaluator.device.type == "cuda":
            model.compile(mode="reduce-overhead")
        return model

    def optimizer(self, model: nn.Module, anchor: Anchor) -> torch.optim.AdamW:
        """AdamW as one fused kernel on the GPU; per-parameter launches were most of a step's device time."""
        optim = self.config.optim
        fused = self.evaluator.device.type == "cuda"
        groups = Trunk.groups(model, anchor.groups(model, optim.weight_decay), optim.lr * optim.pretrained)
        return torch.optim.AdamW(groups, lr=optim.lr, weight_decay=optim.weight_decay, fused=fused)

    def balanced(self, y: Int[Tensor, " n"]) -> Float[Tensor, " v"] | None:
        """Each class's cross-entropy weight, `count ** -balance` over the labelled `y`, mean 1 over the classes
        present and 0 for the absent; None at balance 0, the unweighted loss itself."""
        counts = torch.bincount(y[y != IGNORE], minlength=VOCAB_SIZE).float()
        present = counts > 0
        weight = torch.where(present, counts.clamp(min=1).pow(-self.config.optim.balance), 0.0)
        return weight / weight[present].mean() if self.config.optim.balance else None

    def step(self, state: Step, train: Split, spans: Int[Tensor, " b"]) -> Float[Tensor, ""]:
        """One optimiser step on the words of `spans`; returns the loss without syncing it to the host."""
        batch = train.batch(spans, state.rows, self.config.augment.jitter)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            context, word, *extra = state.model(self.config.augment(batch.x), batch.onset, batch.slot, batch.subject)
            smoothing, objective, y = (
                self.config.optim.label_smoothing,
                self.config.context.objective,
                batch.target(train.y),
            )
            value = objective.loss(
                context.flatten(0, 1), y.flatten(), smoothing, state.weight
            ) + nn.functional.cross_entropy(
                word.flatten(0, 1), y.flatten(), weight=state.weight, ignore_index=IGNORE, label_smoothing=smoothing
            )
            value = value + self.auxiliary.loss(state.model, extra, batch.target(train.token))
        state.optimizer.zero_grad(set_to_none=True)
        value.backward()
        nn.utils.clip_grad_norm_(state.model.parameters(), self.config.optim.clip)
        state.optimizer.step()
        state.anchor.pull(state.model, float(state.schedule.get_last_lr()[0]))
        state.schedule.step()
        state.ema.update(state.model)
        return value.detach()

    def state(self, net: SentenceNet, train: Split, total: int) -> Step:
        """Everything `step` advances to train `net` on `train` for `total` steps: `net` with its auxiliary heads, the
        optimiser and its one-cycle schedule, the anchor, the weight average, and the class weights."""
        optim = self.config.optim
        model = self.auxiliary.trained(net, self.compiled)
        anchor = Anchor.of(optim.anchor, optim.lr, total)
        optimizer = self.optimizer(model, anchor)
        peaks = [group["lr"] for group in optimizer.param_groups]
        schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, peaks, total_steps=total)
        return Step(
            model,
            optimizer,
            schedule,
            train.capacity(optim.batch),
            Ema.of(model, optim.ema),
            self.balanced(train.y),
            anchor,
        )

    def fit(
        self, train: Split, validation: dict[str, Split], log_prior: Float[Tensor, " v"]
    ) -> tuple[nn.Module, list[Epoch]]:
        optim = self.config.optim
        torch.manual_seed(optim.seed)
        net = self.model(train)
        per_epoch = math.ceil(len(train.index) / optim.batch)
        state = self.state(net, train, optim.epochs * per_epoch)
        model = state.model
        history: list[Epoch] = []
        best_state = net.state_dict()
        for epoch in range(optim.epochs):
            model.train()
            weights = Annealed.of(optim.curriculum).weights(train, epoch, optim.epochs)
            draws = torch.multinomial(weights, per_epoch * optim.batch, replacement=True)
            value = torch.stack([self.step(state, train, spans) for spans in draws.split(optim.batch)]).mean().item()
            if not math.isfinite(value):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}")
            with state.ema.applied(model):
                row = self.evaluator.report(net, validation, log_prior, epoch, value)
                history.append(row)
                LOG.info("%s", asdict(row))
                if self.config.selected(history) is row:
                    best_state = {k: v.clone() for k, v in net.state_dict().items()}
        net.load_state_dict(best_state)
        return net, history
