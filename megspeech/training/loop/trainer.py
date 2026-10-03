"""Fit the sentence-context word classifier: fixed-shape optimiser steps, best epoch kept by validation.

Every word is read inside its span (`megspeech.data.spans.spans`): a CNN embeds each 1 s window, a transformer refines
the embeddings across the span, and both the isolated-word head and the context head are trained.
"""

import functools
import logging
import math
import operator
from dataclasses import asdict, dataclass, field

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn

from core.paths import DataRoot
from megspeech.data.batching.split import Split
from megspeech.data.corpus import Corpus
from megspeech.evaluation.scoring.evaluator import Evaluator
from megspeech.evaluation.scoring.score import Epoch
from megspeech.model.input.scaler import SCALER_SAMPLE, Scaler
from megspeech.model.sentence import SentenceNet
from megspeech.objective.context import Objective
from megspeech.objective.targets.t5 import Prototypes
from megspeech.objective.term import ContextTerm, LabelTerm, Term, WordTerm
from megspeech.training.auxiliary import Auxiliary, NoAuxiliary
from megspeech.training.config import RunConfig
from megspeech.training.hook import Scheduled, StepHook
from megspeech.training.loop.ema import Ema
from megspeech.training.regularise.anchor import Anchor
from megspeech.training.sampling.curriculum import Annealed

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainState:
    """What one optimiser step mutates, the terms its loss sums over the model's readouts, and the word rows of its
    fixed-shape batch."""

    model: nn.Module
    optimizer: torch.optim.Optimizer
    schedule: Scheduled
    rows: int
    terms: tuple[Term, ...]
    readouts: tuple[nn.Module, ...]
    ema: Ema
    hooks: tuple[StepHook, ...]


@dataclass
class Trainer:
    config: RunConfig
    evaluator: Evaluator
    prototypes: Float[Tensor, "v d"] | None = None
    auxiliary: Auxiliary = field(default_factory=NoAuxiliary)
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
        groups = anchor.groups(model, optim.weight_decay)
        return torch.optim.AdamW(groups, lr=optim.lr, weight_decay=optim.weight_decay, fused=fused)

    def terms(self, train: Split) -> tuple[Term, ...]:
        """The context head's term, the word head's softmax, then the auxiliary heads'; class-weighted alike."""
        optim = self.config.optim
        classes = LabelTerm.balanced(train.y, train.weight, optim.balance)
        context = ContextTerm(optim.label_smoothing, classes, objective=self.config.context.objective)
        return context, WordTerm(optim.label_smoothing, classes), *self.auxiliary.terms

    def step(self, state: TrainState, train: Split, spans: Int[Tensor, " b"]) -> Float[Tensor, ""]:
        """One optimiser step on the words of `spans`; returns the loss without syncing it to the host."""
        batch = train.batch(spans, state.rows, self.config.augment.jitter)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = state.model(self.config.augment(batch.x), batch.onset, batch.slot, batch.subject)
            parts = zip(state.terms, state.readouts, outputs, strict=True)
            losses = (term.loss(head, output, batch.target(term.labels(train))) for term, head, output in parts)
            value = functools.reduce(operator.add, losses)
        state.optimizer.zero_grad(set_to_none=True)
        value.backward()
        nn.utils.clip_grad_norm_(state.model.parameters(), self.config.optim.clip)
        state.optimizer.step()
        lr = state.schedule.lr()
        for hook in state.hooks:
            hook.step(state.model, lr)
        return value.detach()

    def state(self, net: SentenceNet, train: Split, total: int) -> TrainState:
        """Everything `step` advances to train `net` on `train` for `total` steps: `net` with its auxiliary heads, the
        optimiser and its one-cycle schedule, the anchor, the weight average, and the loss terms."""
        optim = self.config.optim
        model = self.auxiliary.trained(net, self.compiled)
        anchor = Anchor.of(optim.anchor, optim.lr, total)
        optimizer = self.optimizer(model, anchor)
        peaks = [group["lr"] for group in optimizer.param_groups]
        schedule = Scheduled(torch.optim.lr_scheduler.OneCycleLR(optimizer, peaks, total_steps=total))
        ema = Ema.of(model, optim.ema)
        return TrainState(
            model,
            optimizer,
            schedule,
            self.rows(train),
            self.terms(train),
            model.readouts(),
            ema,
            (anchor, schedule, ema),
        )

    def rows(self, train: Split) -> int:
        """Word rows for one batch at the curriculum's draw weights in whichever epoch draws the most words."""
        optim = self.config.optim
        curriculum = Annealed.of(optim.curriculum)
        epochs = range(optim.epochs)
        return max(train.capacity(optim.batch, curriculum.weights(train, epoch, optim.epochs)) for epoch in epochs)

    def fit(
        self, train: Split, validation: dict[str, Split], log_prior: Float[Tensor, " v"]
    ) -> tuple[nn.Module, list[Epoch]]:
        optim = self.config.optim
        torch.manual_seed(optim.seed)
        net = self.model(train)
        per_epoch = math.ceil(len(train.index) / optim.batch)
        state = self.state(net, train, optim.epochs * per_epoch)
        model = state.model
        curriculum = Annealed.of(optim.curriculum)
        history: list[Epoch] = []
        best_state = net.state_dict()
        for epoch in range(optim.epochs):
            model.train()
            weights = curriculum.weights(train, epoch, optim.epochs)
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
