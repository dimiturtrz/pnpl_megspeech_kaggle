"""Self-supervised pretraining of the word encoder by masked latent prediction (data2vec, Baevski et al. 2022; BYOL's
slow target, Grill et al. 2020): a student reads a window with patches of it hidden and predicts what an EMA teacher
of itself reads, at those patches, from the whole window. The loss is a weighted sum of `Objective`s, which add
iBOT's, DINO's and DINOv2's prototype distillation and spread regulariser. The encoder is `SentenceNet`'s own, so the
run it leaves is a `--optim.stem`."""

import copy
import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from functools import reduce
from operator import add
from pathlib import Path
from typing import override

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn
from torch.nn import functional

from core.paths import DataRoot
from megspeech.data.cache.run_cache import DeviceRecording
from megspeech.data.config import METRICS, DataConfig
from megspeech.data.raw.fetch import NEUROMAG
from megspeech.data.raw.masc import Neuromag
from megspeech.data.raw.runs import DEEP, Run
from megspeech.model.config import CONTEXT, ContextConfig, Covariance
from megspeech.model.input.scaler import Scaler
from megspeech.model.sentence import SentenceNet
from megspeech.model.words.conv import ConvWords
from megspeech.model.words.layers.norm import MaskedBatchNorm1d
from megspeech.training.loop.ema import Ema
from megspeech.training.pretrain.masking import Channels, Masking, Region
from megspeech.training.pretrain.objectives import Batch, Kind, Loss, Objective, Readings, Setup, Term
from megspeech.training.pretrain.progress import Probing, Progress
from megspeech.training.pretrain.prototypes import Prototypes
from megspeech.training.pretrain.windows import Pool, Sessions

LOG = logging.getLogger(__name__)

PREDICTOR = "predictor.pt"
MIN_PEERS = 2
"""Windows a batch needs for a term that compares them."""


class CollapseError(RuntimeError):
    """The student's predictions or the teacher's targets stopped varying across the batch."""


@dataclass(frozen=True)
class PretrainConfig:
    data: DataConfig = DataConfig()
    context: ContextConfig = CONTEXT
    masking: Masking = Masking()
    objectives: tuple[Objective, ...] = (Objective(Kind.REGRESSION),)
    """The loss is the sum of these terms, each scaled by its weight. The default is data2vec's regression alone."""
    prototypes: Prototypes = Prototypes()
    epochs: int = 20
    batch: int = 128
    lr: float = 5e-4
    weight_decay: float = 0.05
    clip: float = 1.0
    seed: int = 0
    windows: int = 0
    """Windows one epoch draws; 0 draws as many as the sessions hold non-overlapping, so each sample is seen once on
    average."""
    sessions: int = 0
    """Only the first this-many sessions (name order) are read, for a smoke; 0 reads every one."""
    store: str = "cpu"
    """Where the sessions' MEG lives: all of subject 0's training sessions are ~9 GB, more than a shared GPU holds."""
    fit: int = 10000
    """Random windows the scaler and whitener are fit on."""
    ema: float = 0.999
    """The teacher's decay once ramped."""
    ema_start: float = 0.99
    ema_ramp: float = 0.3
    """Share of the steps over which the decay rises linearly from `ema_start` to `ema`."""
    top: int = 3
    """Last blocks whose layer-normalised feature maps the target averages."""
    loss: Loss = Loss.SMOOTH_L1
    """The regression objective's distance."""
    monitor: int = 200
    """Steps between collapse checks."""
    collapse_floor: float = 0.05
    """Spread (1 isotropic, 0 collapsed) under which the run aborts."""
    teacher: bool = False
    """Keep the teacher's weights, not the student's."""
    probe: Probing = Probing()
    """The frozen-encoder k-NN probe on carved-val words that tracks progress, which the loss cannot."""

    def __post_init__(self) -> None:
        if self.context.covariance is Covariance.NOISE:
            raise ValueError("a noise covariance is fit on labelled residuals, which pretraining never reads")
        if not 0 < self.top <= len(self.context.conv.dilations):
            raise ValueError(f"top {self.top} is not among the {len(self.context.conv.dilations)} blocks")
        kinds = [objective.kind for objective in self.objectives]
        if not kinds or len(set(kinds)) != len(kinds):
            raise ValueError(f"objectives {kinds} must be a non-empty list of distinct kinds")
        if self.batch < MIN_PEERS and {Kind.DINO, Kind.KOLEO} & set(kinds):
            raise ValueError("cross-view and KoLeo terms compare windows in a batch")
        if not self.ema_start <= self.ema < 1:
            raise ValueError(f"ema {self.ema} must be in [ema_start {self.ema_start}, 1)")


class Predictor(nn.Module):
    """The student's readout of its feature map at each time step, and the learned vector standing for a masked one."""

    def __init__(self, channels: int, hidden: int) -> None:
        super().__init__()
        self.token = nn.Parameter(torch.zeros(channels, 1))
        self.body = nn.Sequential(nn.Conv1d(hidden, hidden, 1), nn.GELU(), nn.Conv1d(hidden, hidden, 1))

    @override
    def forward(self, maps: Float[Tensor, "b h t"]) -> Float[Tensor, "b t h"]:
        return self.body(maps).transpose(1, 2)


@dataclass(frozen=True)
class Student:
    """What a step trains besides the objectives' heads: the encoder and the predictor that reads it."""

    net: SentenceNet
    predictor: Predictor


@dataclass(frozen=True)
class Target:
    """The EMA teacher: a copy of the student that reads the clean window, its batch statistics normalising as the
    student's do, its dropout off."""

    net: SentenceNet
    top: int

    @classmethod
    def of(cls, net: SentenceNet, top: int) -> "Target":
        teacher = copy.deepcopy(net).eval().requires_grad_(requires_grad=False)
        for module in teacher.modules():
            if isinstance(module, MaskedBatchNorm1d):
                module.train()
        return cls(teacher, top)

    @staticmethod
    def convolutional(net: SentenceNet) -> ConvWords:
        """`net`'s word encoder, which must keep a per-timestep feature map."""
        if not isinstance(net.words, ConvWords):
            raise TypeError(f"{type(net.words).__name__} has no per-timestep feature map")
        return net.words

    @torch.no_grad()
    def follow(self, net: nn.Module, decay: float) -> None:
        """Move the teacher `1 - decay` of the way to `net`."""
        Ema(decay, Ema.state(self.net)).step(net, 0.0)

    @torch.no_grad()
    def maps(self, shown: Float[Tensor, "b c t"]) -> list[Float[Tensor, "b h t"]]:
        """The feature map of each block over `shown`, a window as the encoder reads it."""
        return self.convolutional(self.net).taps(shown)

    @torch.no_grad()
    def prepare(self, x: Float[Tensor, "b c t"], subject: Int[Tensor, " b"]) -> Float[Tensor, "b c t"]:
        return self.net.prepare(x, subject)

    def target(self, maps: list[Float[Tensor, "b h t"]]) -> Float[Tensor, "b t h"]:
        """The mean over the last `top` of `maps`, each layer-normalised at every time step."""
        return torch.stack(
            [functional.layer_norm(m.transpose(1, 2).float(), m.shape[1:2]) for m in maps[-self.top :]]
        ).mean(0)


@dataclass(frozen=True)
class Round:
    epoch: int
    loss: float
    """The weighted total, as a mean over the epoch's steps."""
    stats: dict[str, float]
    """The epoch's mean of each term's loss (`loss/<term>`) and each diagnostic it read."""
    seconds: float
    probe: dict[str, float] = field(default_factory=dict)
    """The frozen-encoder probe's BAcc@10 of the student and teacher after the epoch, when read. Epoch -1 is the
    random-init floor, read before any step."""


@dataclass
class Pretrainer:
    config: PretrainConfig
    device: torch.device
    pool: Pool
    region: Region = field(default_factory=Region)
    progress: Progress | None = None

    def __post_init__(self) -> None:
        if (
            self.config.masking.channels is Channels.REGION
            and self.region.nearest.shape[0] != self.pool.recording.meg.shape[0]
        ):
            raise ValueError("region masking needs the sensor positions of every recorded channel")

    @classmethod
    def of(cls, config: PretrainConfig, root: DataRoot, device: torch.device) -> "Pretrainer":
        """A pretrainer over `config.sessions`' MEG, read only from sessions `Sessions.of` clears."""
        data = config.data
        sessions = Sessions.of(data, Run.everywhere(root)).limited(config.sessions)
        where = torch.device(config.store)
        recording = DeviceRecording.load(
            data.run_cache(root),
            where,
            sessions.names,
            geometry=data.geometry(root, where),
        )
        LOG.info("pretraining on %d sessions, %.2f GB on %s", len(sessions.runs), recording.gigabytes, where)
        progress = None
        if config.probe.every:
            progress = Progress.load(data, root, recording, config.probe, device)
        return cls(config, device, Pool(recording, data.samples), cls.sensors(config, root, device), progress)

    @staticmethod
    def sensors(config: PretrainConfig, root: DataRoot, device: torch.device) -> Region:
        """The Neuromag-306 neighbourhoods region masking silences; none for another channel strategy."""
        if config.masking.channels is not Channels.REGION:
            return Region()
        positions = torch.from_numpy(Neuromag.of(NEUROMAG.home(root)).positions).float().to(device)
        return Region.of(positions, config.masking.region)

    def net(self, generator: torch.Generator) -> SentenceNet:
        """A fresh model whose scaler and whitener are fit on random windows."""
        sample = self.pool.sample(self.config.fit, generator).to(self.device)
        model = SentenceNet(Scaler(sample), context=self.config.context, channels=sample.shape[1]).to(self.device)
        model.whiten.fit(model.scale(sample))
        return model

    def decay(self, step: int, steps: int) -> float:
        """The teacher's decay at `step` of `steps`."""
        config = self.config
        ramp = min(1.0, step / max(1.0, config.ema_ramp * steps))
        return config.ema_start + (config.ema - config.ema_start) * ramp

    def show(
        self,
        student: Student,
        x: Float[Tensor, "b c t"],
        subject: Int[Tensor, " b"],
        generator: torch.Generator,
    ) -> tuple[Float[Tensor, "b c t"], Bool[Tensor, "b t"]]:
        """One masked view of `x`: time patches to hide drawn, sensors silenced on the raw window if the strategy
        says so, then the model's own preparation, then the patches hidden behind the predictor's token; and the
        hidden samples."""
        masking = self.config.masking
        hidden = masking.draw(len(x), x.shape[-1], self.config.data.rate, generator).to(self.device)
        prepared = student.net.prepare(masking.silence(x, self.region, generator), subject)
        return masking.apply(prepared, hidden, student.predictor.token, generator), hidden

    def step(
        self,
        student: Student,
        teacher: Target,
        terms: list[Term],
        generator: torch.Generator,
        progress: float = 0.0,
    ) -> Readings:
        """One window batch's weighted loss, each term's and its diagnostics, not yet synced to the host; the
        gradients are left on the parameters."""
        config = self.config
        x = self.pool.sample(config.batch, generator).to(self.device)
        subject = torch.full((config.batch,), DEEP, dtype=torch.long, device=self.device)
        with torch.autocast(self.device.type, dtype=torch.bfloat16):
            clean = teacher.maps(teacher.prepare(x, subject))
            shown, hidden = zip(
                *(self.show(student, x, subject, generator) for _ in range(max(t.views for t in terms))),
                strict=True,
            )
            convolutional = teacher.convolutional(student.net)
            taps = [convolutional.taps(view)[-1] for view in shown]
            across = [teacher.maps(view)[-1] for view in shown] if any(t.cross for t in terms) else []
            batch = Batch(taps, list(hidden), clean[-1], teacher.target(clean), across, progress)
            scored = [(term, *term.loss(batch)) for term in terms]
        total = reduce(add, (term.weight * loss for term, loss, _ in scored))
        self.clear(student, terms)
        total.backward()
        losses = {f"loss/{term.name}": loss.detach() for term, loss, _ in scored}
        return {"loss": total.detach(), **losses, **{k: v.detach() for _, _, read in scored for k, v in read.items()}}

    @staticmethod
    def clear(student: Student, terms: list[Term]) -> None:
        """Drop every gradient the step is about to recompute."""
        student.net.zero_grad(set_to_none=True)
        student.predictor.zero_grad(set_to_none=True)
        for term in terms:
            for parameter in term.parameters():
                parameter.grad = None

    @staticmethod
    def average(stats: list[Readings]) -> dict[str, float]:
        """The mean over steps of each reading, synced to the host once."""
        keys = list(stats[0])
        means = torch.stack([torch.stack([s[k] for k in keys]) for s in stats]).mean(0)
        return dict(zip(keys, means.tolist(), strict=True))

    @staticmethod
    def watch(stats: dict[str, float], floors: dict[str, float]) -> None:
        """Raise when `stats` (a mean over steps) show a non-finite loss or a diagnostic under its floor."""
        if not math.isfinite(stats["loss"]):
            raise FloatingPointError("non-finite pretraining loss")
        collapsed = {name: stats[name] for name, floor in floors.items() if stats[name] < floor}
        if collapsed:
            raise CollapseError(f"under their floors {floors}: {collapsed}")

    def probed(self, net: SentenceNet, teacher: Target, epoch: int) -> dict[str, float]:
        """The probe's readings after `epoch` if it is due, every `probe.every` epochs and the last; else none."""
        every = self.config.probe.every
        if not self.progress or not every or not ((epoch + 1) % every == 0 or epoch + 1 == self.config.epochs):
            return {}
        return self.progress.read(net, teacher.net)

    def fit(self, net: SentenceNet, predictor: Predictor, generator: torch.Generator) -> tuple[Target, list[Round]]:
        """Train `net`, `predictor` and the objectives' heads; returns the teacher and each epoch's record."""
        config = self.config
        teacher = Target.of(net, config.top)
        setup = Setup(
            predictor, config.context.conv.hidden, config.prototypes, config.loss, config.collapse_floor, self.device
        )
        terms = [objective.build(setup) for objective in config.objectives]
        floors = {name: floor for term in terms for name, floor in term.floors().items()}
        per_epoch = math.ceil((config.windows or self.pool.total // self.pool.length) / config.batch)
        steps = per_epoch * config.epochs
        params = [*net.encoder(), *predictor.parameters(), *(p for term in terms for p in term.parameters())]
        optimizer = torch.optim.AdamW(
            params, lr=config.lr, weight_decay=config.weight_decay, fused=self.device.type == "cuda"
        )
        schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, config.lr, total_steps=steps)
        history: list[Round] = []
        if self.progress:
            history.append(Round(-1, math.nan, {}, 0.0, self.progress.read(net, teacher.net)))
            LOG.info("%s", asdict(history[-1]))
        for epoch in range(config.epochs):
            net.train()
            began = time.perf_counter()
            stats: list[Readings] = []
            for step in range(per_epoch):
                done = epoch * per_epoch + step
                stats.append(self.step(Student(net, predictor), teacher, terms, generator, done / steps))
                nn.utils.clip_grad_norm_(params, config.clip)
                optimizer.step()
                schedule.step()
                decay = self.decay(done, steps)
                teacher.follow(net, decay)
                for term in terms:
                    term.follow(decay)
                if (step + 1) % config.monitor == 0:
                    self.watch(self.average(stats[-config.monitor :]), floors)
            mean = self.average(stats)
            self.watch(mean, floors)
            seconds = time.perf_counter() - began
            history.append(Round(epoch, mean["loss"], mean, seconds, self.probed(net, teacher, epoch)))
            LOG.info("%s", asdict(history[-1]))
        return teacher, history

    def run(self) -> tuple[SentenceNet, Predictor, list[Round]]:
        """Pretrain a fresh model; returns the kept one (the teacher's when `config.teacher`)."""
        torch.manual_seed(self.config.seed)
        generator = torch.Generator().manual_seed(self.config.seed)
        net = self.net(generator)
        predictor = Predictor(self.pool.recording.meg.shape[0], self.config.context.conv.hidden).to(self.device)
        teacher, history = self.fit(net, predictor, generator)
        return (teacher.net if self.config.teacher else net), predictor, history

    def save(self, directory: Path, net: SentenceNet, predictor: Predictor, history: list[Round]) -> None:
        """The run `--optim.stem` loads: the model's state, its context config, and the record."""
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(net.state_dict(), directory / "model.pt")
        torch.save(predictor.state_dict(), directory / PREDICTOR)
        context = self.config.context
        context.save(directory)
        record = {"config": asdict(self.config), "history": [asdict(row) for row in history]}
        (directory / METRICS).write_text(json.dumps(record, indent=1, default=str))
