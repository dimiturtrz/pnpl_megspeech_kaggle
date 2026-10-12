"""Everything a training run is made of, and the record a finished run leaves of it."""

import json
from dataclasses import dataclass
from pathlib import Path

from core.rules import Rule
from core.tree import Tree
from megspeech.data.config import METRICS, DataConfig, Refit
from megspeech.evaluation.scoring.score import Epoch, SelectionConfig
from megspeech.evaluation.scoring.telemetry import TelemetryConfig
from megspeech.model.config import CONTEXT, ContextConfig, Encoder
from megspeech.model.words.trunk import RATE as TRUNK_RATE
from megspeech.objective.targets.loudness import FRAMES
from megspeech.submission.config import ReportConfig
from megspeech.training.optim import OptimConfig
from megspeech.training.regularise.augment import Augment
from megspeech.training.sampling.sampler import SamplerConfig


@dataclass(frozen=True)
class StopConfig:
    """When the epoch loop ends before its schedule does."""

    patience: int = 0
    """Stop once the selection metric (macro bacc10) has not strictly improved for this many epochs, keeping the best
    epoch as ever; 0 trains every epoch. The learning-rate schedule stays defined over all `epochs`, so a stopped run
    is that schedule truncated. Refused where no epoch is selected on validation (`refit`, `last`, no heads)."""

    def stalled(self, history: list[Epoch]) -> bool:
        """Whether `history`'s best epoch (the first of equals) is `patience` or more epochs behind its last."""
        best = max(range(len(history)), key=lambda at: history[at].macro.bacc10)
        return self.patience > 0 and len(history) - 1 - best >= self.patience


@dataclass(frozen=True)
class RunConfig:
    """Everything a training run is made of; `core.tree.Tree` turns it into the command line."""

    data: DataConfig = DataConfig()
    context: ContextConfig = CONTEXT
    augment: Augment = Augment()
    optim: OptimConfig = OptimConfig()
    sampler: SamplerConfig = SamplerConfig()
    selection: SelectionConfig = SelectionConfig()
    stop: StopConfig = StopConfig()
    report: ReportConfig = ReportConfig()
    telemetry: TelemetryConfig = TelemetryConfig()

    def __post_init__(self) -> None:
        Rule.enforce(self, RUN_RULES)

    @property
    def keeps_last(self) -> bool:
        """Whether the run keeps its last epoch: val was trained on, or the heads it would select on are not."""
        return self.data.refit is not Refit.NONE or self.optim.last or not self.optim.heads

    def selected(self, history: list[Epoch]) -> Epoch:
        """The epoch whose weights the run keeps: the best on validation, or the last when val was trained on."""
        return history[-1] if self.keeps_last else max(history, key=lambda row: row.macro.bacc10)

    def stalled(self, history: list[Epoch]) -> bool:
        """Whether the run stops early after `history`."""
        return self.stop.stalled(history)

    @classmethod
    def saved(cls, directory: Path) -> "RunConfig":
        """The config a run in `directory` recorded, flat records from before the tree included, its data section
        under the retired `extra`/`share`/`stratify` fields migrated to `val` and `refit`, its sampler knobs from
        `optim` to `sampler`, its flat auxiliary, adversary, teacher and transfer knobs into their `optim` groups."""
        return Tree(cls).load(json.loads((directory / METRICS).read_text())["config"], saved=True)


RUN_RULES: tuple[Rule[RunConfig], ...] = (
    Rule(
        lambda r: not 0 <= r.optim.topk <= 1 or r.optim.topk_temperature <= 0 or r.optim.topk_margin < 0,
        lambda r: (
            f"optim.topk {r.optim.topk} needs to be in [0, 1], topk_temperature {r.optim.topk_temperature} above 0 "
            f"and topk_margin {r.optim.topk_margin} at least 0"
        ),
    ),
    Rule(
        lambda r: bool(r.optim.fused_lr) and len(r.optim.fused_lr) != len(r.context.fused),
        lambda r: f"fused_lr {r.optim.fused_lr} gives a rate for each of the fused {r.context.fused}",
    ),
    Rule(
        lambda r: r.optim.auxiliaries.listens and r.context.encoder is not Encoder.CONV,
        lambda r: f"optim.acoustic/audio read the conv encoder's time features, not encoder {r.context.encoder}",
    ),
    Rule(
        lambda r: r.optim.auxiliaries.listens and r.context.conv.pyramid.active,
        lambda _: "optim.acoustic/audio read the encoder's time course beside the heads, which the pyramid corrects",
    ),
    Rule(
        lambda r: r.optim.auxiliaries.listens and r.data.rate != FRAMES,
        lambda r: (
            f"optim.acoustic/audio cut the {FRAMES} Hz envelope in the cache's frames, but data.decimate "
            f"{r.data.decimate} reads at {r.data.rate} Hz"
        ),
    ),
    Rule(
        lambda r: r.context.encoder is Encoder.SPANNED and r.data.rate != TRUNK_RATE,
        lambda r: (
            f"the spanned encoder's MEG-XL trunk reads {TRUNK_RATE} Hz, but data.decimate {r.data.decimate} "
            f"reads at {r.data.rate} Hz"
        ),
    ),
    Rule(
        lambda r: r.stop.patience < 0,
        lambda r: f"stop.patience {r.stop.patience} is negative; 0 trains every epoch",
    ),
    Rule(
        lambda r: bool(r.stop.patience) and r.keeps_last,
        lambda r: (
            f"stop.patience {r.stop.patience} needs an epoch selected on validation, but this run keeps its "
            "last (refit, optim.last, or heads 0)"
        ),
    ),
    Rule(
        lambda r: bool(r.data.projected) and bool(r.context.encoder.cached or r.context.conv.merger),
        lambda r: (
            f"encoder {r.context.encoder} with merger {r.context.conv.merger} reads the 306 sensors, which "
            "a projected config does not keep"
        ),
    ),
    Rule(
        lambda r: (r.context.encoder is Encoder.SPANNED) != bool(r.data.segment),
        lambda r: (
            f"the spanned encoder, and it alone, reads continuous segments: encoder {r.context.encoder}, "
            f"data.segment {r.data.segment}"
        ),
    ),
    Rule(
        lambda r: bool(r.data.segment) and bool(r.augment.bgmix or r.augment.average > 1 or r.augment.jitter),
        lambda _: "bgmix, group averaging and jitter act on word windows, which a segment batch does not hold",
    ),
    Rule(
        lambda r: bool(r.data.segment) and bool(r.augment.right_cut),
        lambda _: "a right cut drops words a segment batch still reads in its span's continuous signal",
    ),
)
