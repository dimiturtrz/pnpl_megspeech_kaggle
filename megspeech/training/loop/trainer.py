"""Fit the sentence-context word classifier: fixed-shape optimiser steps, best epoch kept by validation.

Every word is read inside its span (`megspeech.data.spans.spans`): a CNN embeds each 1 s window, a transformer refines
the embeddings across the span, and both the isolated-word head and the context head are trained.
"""

import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import cast

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn

from core.paths import DataRoot
from megspeech.data.batching.split import Split
from megspeech.data.corpus import Corpus
from megspeech.data.labels.vocabulary import IGNORE
from megspeech.device import Device
from megspeech.evaluation.scoring.evaluator import Evaluator
from megspeech.evaluation.scoring.score import Epoch
from megspeech.evaluation.scoring.telemetry import Telemetry
from megspeech.model.branch import Branch
from megspeech.model.fusion import Fusion
from megspeech.model.input.scaler import SCALER_SAMPLE, Scaler
from megspeech.model.input.subjects import Unseen
from megspeech.model.sentence import SentenceNet
from megspeech.objective.anchors import Anchors
from megspeech.objective.context import Objective
from megspeech.objective.information import InformedTerm
from megspeech.objective.targets.t5 import Prototypes
from megspeech.objective.term import ContextTerm, LabelTerm, Term, WordTerm
from megspeech.objective.topk import TopKMix
from megspeech.training.auxiliary import Auxiliary, NoAuxiliary
from megspeech.training.config import RunConfig
from megspeech.training.hook import Scheduled, StepHook
from megspeech.training.loop.ema import Ema
from megspeech.training.loop.gradnorm import GradNorms
from megspeech.training.ramp import Ramp
from megspeech.training.regularise.anchor import Anchor
from megspeech.training.regularise.average import GroupAverage
from megspeech.training.regularise.bgmix import BGMix, Untouched
from megspeech.training.regularise.noise import SAMPLES
from megspeech.training.regularise.regional import Regional
from megspeech.training.regularise.rightcut import RightCut, Whole
from megspeech.training.sampling.curriculum import Curriculum, Natural
from megspeech.training.sampling.pairing import Pairing, Unpaired
from megspeech.training.sampling.sampler import Sampler
from megspeech.training.teacher import Teacher, Untaught
from megspeech.training.timing_expert import TimingExpert

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
    curriculum: Curriculum = field(default_factory=Natural)
    bgmix: Untouched = field(default_factory=Untouched)
    average: Untouched = field(default_factory=Untouched)
    grads: GradNorms = field(default_factory=GradNorms)
    cut: Whole = field(default_factory=Whole)


@dataclass
class Trainer:
    config: RunConfig
    evaluator: Evaluator
    prototypes: Float[Tensor, "v d"] | None = None
    auxiliary: Auxiliary = field(default_factory=NoAuxiliary)
    initial: dict[str, Tensor] = field(default_factory=dict)
    teacher: Teacher = field(default_factory=Untaught)
    regional: Regional = field(default_factory=Regional.off)
    timing: dict[str, Tensor] = field(default_factory=dict)
    """The frozen timing expert's head weights, when the run trains over one."""
    heads: dict[str, Tensor] = field(default_factory=dict)
    """The auxiliary heads' weights at the kept epoch, which the model's own state does not hold."""
    stem: dict[str, Tensor] = field(default_factory=dict)
    """The trained run's state whose encoder the model starts from, when the run names one."""
    fused: tuple[Path, ...] = ()
    """The saved runs whose encoders the model reads beside its own (`ContextConfig.fused`)."""
    pairing: Unpaired = field(default_factory=Unpaired)
    """What writes cross-subject pairs into each batch's draws (`SamplerConfig.pairs`)."""

    @staticmethod
    def corpus(config: RunConfig, root: DataRoot, device: torch.device) -> Corpus:
        """The data `config` records, on `device`."""
        return Corpus.load(config.data, root, device)

    @classmethod
    def of(cls, corpus: Corpus, config: RunConfig, root: DataRoot) -> "Trainer":
        """A trainer on `corpus`'s device, selecting on its unseen spans; SigLIP reads the text prototypes, a warm
        start its run's weights."""
        siglip = config.context.objective is Objective.SIGLIP
        prototypes = Prototypes.load(Prototypes.path(root)).vectors if siglip else None
        anchored = Anchors.of(config.context.anchor, root, corpus.positions, corpus.train.y)
        if anchored is not None:
            anchored.report(config.context.anchor)
            prototypes = anchored.reduce(config.context.anchor_dim).vectors
        device = corpus.train.y.device
        evaluator = Evaluator(device, corpus.seen, config.selection.min_novel)
        warm = config.optim.warm
        initial = torch.load(root.models() / warm / "model.pt", map_location=device, weights_only=True) if warm else {}
        auxiliary = Auxiliary.of(config.optim, root, device, corpus.positions)
        regional = Regional.of(config.augment.spatial, root, device)
        timing = TimingExpert.load(root, config.optim.timing)
        stem = config.optim.stem
        stemmed = torch.load(root.models() / stem / "model.pt", map_location=device, weights_only=True) if stem else {}
        taught = Teacher.of(config.optim, root, config.data, corpus.positions)
        fused = tuple(root.models() / name for name in config.context.fused)
        pairing = Pairing.of(config.sampler.pairs, corpus.positions, corpus.train, config.optim.batch)
        return cls(
            config,
            evaluator,
            prototypes,
            auxiliary,
            initial,
            taught,
            regional,
            timing,
            stem=stemmed,
            fused=fused,
            pairing=pairing,
        )

    def model(self, train: Split) -> SentenceNet:
        """A fresh model, or the warm start's weights when there is one."""
        sample = train.sample(SCALER_SAMPLE)
        context = self.config.context
        channels = sample.shape[1]
        model = SentenceNet(Scaler(sample), context=context, prototypes=self.prototypes, channels=channels)
        model.fuse(Fusion(Branch.many(self.fused, context, channels), context.branch_drop))
        model.whiten.fit(model.scale(SAMPLES[self.config.context.covariance].windows(train, sample)))
        if self.initial:
            model.warm(self.initial)
        if self.stem:
            model.adopt(self.stem)
        if self.timing:
            model.timing.frozen(self.timing)
        return self.compiled(model.to(self.evaluator.device))

    def compiled[M: nn.Module](self, model: M) -> M:
        """`model` compiled in place into CUDA graphs when the run compiles and is on the GPU."""
        if self.config.optim.compiled and self.evaluator.device.type == Device.CUDA:
            model.compile(mode="reduce-overhead")
        return model

    def optimizer(
        self,
        model: nn.Module,
        anchor: Anchor,
        encoder: tuple[nn.Parameter, ...] = (),
        branches: tuple[list[nn.Parameter], ...] = (),
    ) -> torch.optim.AdamW:
        """AdamW as one fused kernel on the GPU; per-parameter launches were most of a step's device time. Each of the
        anchor's groups splits off its `encoder` parameters at `encoder_lr` times the rate, and each fused branch's at
        its `fused_lr`'s."""
        optim = self.config.optim
        fused = self.evaluator.device.type == Device.CUDA
        shares = {id(p): optim.encoder_lr for p in encoder}
        for params, share in zip(branches, optim.fused_lr or (1.0,) * len(branches), strict=True):
            shares |= {id(p): share for p in params}
        kept: list[dict[str, object]] = []
        for group in anchor.groups(model, optim.weight_decay):
            by_rate: dict[float, list[nn.Parameter]] = {}
            for p in cast("list[nn.Parameter]", group["params"]):
                by_rate.setdefault(optim.lr * shares.get(id(p), 1.0), []).append(p)
            kept.extend({**group, "params": part, "lr": lr} for lr, part in by_rate.items())
        LOG.info(
            "optimiser groups (lr, params) %s", [(g["lr"], len(cast("list[nn.Parameter]", g["params"]))) for g in kept]
        )
        return torch.optim.AdamW(kept, lr=optim.lr, weight_decay=optim.weight_decay, fused=fused)

    def terms(self, train: Split) -> tuple[Term, ...]:
        """The context head's term as the teacher teaches it, the word head's softmax, then the auxiliary heads';
        class-weighted alike."""
        optim = self.config.optim
        classes = LabelTerm.balanced(train.y, train.weight, optim.balance)
        topk = TopKMix(optim.topk, optim.topk_temperature, optim.topk_margin)
        context = ContextTerm(
            optim.label_smoothing, classes, objective=self.config.context.objective, weight=optim.heads, topk=topk
        )
        context = InformedTerm.of(
            self.teacher.taught(context, train, self.evaluator), optim.mi, train.log_prior().exp()
        )
        return (
            context,
            WordTerm(optim.label_smoothing, classes, weight=optim.heads, topk=topk),
            *self.auxiliary.terms,
        )

    def step(self, state: TrainState, train: Split, spans: Int[Tensor, " b"]) -> Float[Tensor, " k"]:
        """One optimiser step on the words of `spans`; returns each term's loss without syncing it to the host."""
        batch = state.cut.apply(train.batch(spans, state.rows, self.config.augment.jitter), train)
        with torch.autocast(Device.CUDA, dtype=torch.bfloat16):
            mixed = state.bgmix.apply(state.average.apply(batch.x, batch, train), batch, train)
            outputs = state.model(batch.inputs(self.regional(self.config.augment(mixed))))
            parts = zip(state.terms, state.readouts, outputs, strict=True)
            inside = batch.target(train.y) != IGNORE
            losses = torch.stack(
                [term.loss(head, out, batch.target(term.labels(train)), inside) for term, head, out in parts]
            )
            value = losses.sum()
        state.curriculum.observe(spans, outputs[0], batch.target(train.y))
        state.optimizer.zero_grad(set_to_none=True)
        value.backward()
        state.grads.record(nn.utils.clip_grad_norm_(state.model.parameters(), self.config.optim.clip))
        state.optimizer.step()
        lr = state.schedule.lr()
        for hook in state.hooks:
            hook.step(state.model, lr)
        return losses.detach()

    def state(self, net: SentenceNet, train: Split, total: int) -> TrainState:
        """Everything `step` advances to train `net` on `train` for `total` steps: `net` with its auxiliary heads, the
        optimiser and its one-cycle schedule, the anchor, the weight average, and the loss terms."""
        optim = self.config.optim
        model = self.auxiliary.trained(net, self.compiled)
        anchor = Anchor.of(optim.anchor, optim.lr, total, Unseen(optim.anchor_toward))
        optimizer = self.optimizer(model, anchor, tuple(net.encoder()), net.fusion.encoders())
        peaks = [group["lr"] for group in optimizer.param_groups]
        schedule = Scheduled(torch.optim.lr_scheduler.OneCycleLR(optimizer, peaks, total_steps=total))
        ema = Ema.of(model, optim.ema)
        ramp = Ramp.of(model, optim.adversary, round(total * optim.adversary_ramp / optim.epochs))
        return TrainState(
            model,
            optimizer,
            schedule,
            self.rows(train),
            self.terms(train),
            model.readouts(),
            ema,
            (anchor, schedule, ema, ramp),
            Sampler.of(self.config.sampler, train),
            BGMix.of(train, self.config.augment.bgmix, self.config.augment.bgmix_key),
            GroupAverage.of(train, self.config.augment.average, self.config.augment.average_chance),
            GradNorms(tuple(net.encoder())),
            RightCut.of(train, self.config.augment.right_cut, self.config.data.rate),
        )

    def rows(self, train: Split) -> int:
        """Word rows for one batch at the curriculum's draw weights in whichever epoch draws the most words."""
        optim = self.config.optim
        curriculum = Sampler.scheduled(self.config.sampler, train)
        epochs = range(optim.epochs)
        return max(train.capacity(optim.batch, curriculum.weights(train, epoch, optim.epochs)) for epoch in epochs)

    @staticmethod
    def auxiliary_state(model: nn.Module) -> dict[str, Tensor]:
        """The weights of `model`'s auxiliary heads (`WithAuxiliary.heads`), none for a model without."""
        return {k: v for k, v in model.state_dict().items() if k.startswith("heads.")}

    def fit(
        self, train: Split, validation: dict[str, Split], log_prior: Float[Tensor, " v"]
    ) -> tuple[nn.Module, list[Epoch]]:
        optim = self.config.optim
        torch.manual_seed(optim.seed)
        net = self.model(train)
        per_epoch = math.ceil(len(train.index) / optim.batch)
        state = self.state(net, train, optim.epochs * per_epoch)
        model = state.model
        curriculum = state.curriculum
        LOG.info("sampler %s", type(curriculum).__name__)
        history: list[Epoch] = []
        best_state = net.state_dict()
        self.heads = self.auxiliary_state(model)
        telemetry = Telemetry.of(self.evaluator, self.config.telemetry, train, validation)
        for epoch in range(optim.epochs):
            model.train()
            weights = curriculum.weights(train, epoch, optim.epochs)
            if summary := curriculum.summary(train, epoch, optim.epochs, weights):
                LOG.info("epoch %d draw weights against base %s", epoch, summary)
            draws = torch.multinomial(weights, per_epoch * optim.batch, replacement=True)
            draws = self.pairing.drawn(draws, weights, optim.batch)
            terms = torch.stack([self.step(state, train, spans) for spans in draws.split(optim.batch)]).mean(0)
            value = terms.sum().item()
            LOG.info(
                "epoch %d loss by term %s",
                epoch,
                {type(t).__name__: v for t, v in zip(state.terms, terms.tolist(), strict=True)},
            )
            LOG.info("epoch %d grad norms %s", epoch, state.grads.drained())
            LOG.info("epoch %d bgmix mixed words %d", epoch, state.bgmix.drained())
            LOG.info("epoch %d averaged words %d", epoch, state.average.drained())
            LOG.info("epoch %d right cut %s", epoch, state.cut.drained())
            if not math.isfinite(value):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}")
            with state.ema.applied(model):
                row = telemetry.measured(
                    net, self.evaluator.report(net, validation, log_prior, epoch, value), log_prior
                )
                history.append(row)
                LOG.info("%s", asdict(row))
                if self.config.selected(history) is row:
                    best_state = {k: v.clone() for k, v in net.state_dict().items()}
                    self.heads = {k: v.clone() for k, v in self.auxiliary_state(model).items()}
            if self.config.stalled(history):
                best = self.config.selected(history)
                LOG.info("stopped early at epoch %d (best %d)", epoch, best.epoch)
                break
        net.load_state_dict(best_state)
        return net, history
