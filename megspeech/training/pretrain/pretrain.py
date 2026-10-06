"""Self-supervised pretraining of the word encoder by masked latent prediction (data2vec, Baevski et al. 2022; BYOL's
slow target, Grill et al. 2020): a student reads a window with patches of it hidden and predicts what an EMA teacher
of itself reads, at those patches, from the whole window. The encoder is `SentenceNet`'s own, so the run it leaves is a
`--optim.stem`."""

import copy
import json
import logging
import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import override

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.nn import functional

from core.paths import DataRoot
from megspeech.data.cache.run_cache import DeviceRecording
from megspeech.data.config import METRICS, DataConfig
from megspeech.data.raw.runs import DEEP, Run
from megspeech.model.config import CONTEXT, ContextConfig, Covariance
from megspeech.model.input.scaler import Scaler
from megspeech.model.sentence import SentenceNet
from megspeech.model.words.conv import ConvWords
from megspeech.model.words.layers.norm import MaskedBatchNorm1d
from megspeech.training.loop.ema import Ema
from megspeech.training.pretrain.masking import Masking
from megspeech.training.pretrain.windows import Pool, Sessions

LOG = logging.getLogger(__name__)

PREDICTOR = "predictor.pt"


class Loss(StrEnum):
    SMOOTH_L1 = "smooth_l1"
    COSINE = "cosine"


LOSSES: dict[Loss, Callable[[Tensor, Tensor], Tensor]] = {
    Loss.SMOOTH_L1: functional.smooth_l1_loss,
    Loss.COSINE: lambda pred, target: (1 - functional.cosine_similarity(pred, target, dim=-1)).mean(),
}


class CollapseError(RuntimeError):
    """The student's predictions or the teacher's targets stopped varying across the batch."""


@dataclass(frozen=True)
class PretrainConfig:
    data: DataConfig = DataConfig()
    context: ContextConfig = CONTEXT
    masking: Masking = Masking()
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
    monitor: int = 200
    """Steps between collapse checks."""
    collapse_floor: float = 0.05
    """Spread (1 isotropic, 0 collapsed) under which the run aborts."""
    teacher: bool = False
    """Keep the teacher's weights, not the student's."""

    def __post_init__(self) -> None:
        if self.context.covariance is Covariance.NOISE:
            raise ValueError("a noise covariance is fit on labelled residuals, which pretraining never reads")
        if not 0 < self.top <= len(self.context.conv.dilations):
            raise ValueError(f"top {self.top} is not among the {len(self.context.conv.dilations)} blocks")
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
    def read(self, x: Float[Tensor, "b c t"], subject: Int[Tensor, " b"]) -> Float[Tensor, "b t h"]:
        """The mean over the last `top` blocks of each one's feature map, layer-normalised at every time step."""
        maps = self.convolutional(self.net).taps(self.net.prepare(x, subject))[-self.top :]
        return torch.stack([functional.layer_norm(m.transpose(1, 2).float(), m.shape[1:2]) for m in maps]).mean(0)


@dataclass(frozen=True)
class Round:
    epoch: int
    loss: float
    spread: float
    target_spread: float
    target_centre: float
    seconds: float


@dataclass
class Pretrainer:
    config: PretrainConfig
    device: torch.device
    pool: Pool

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
            standardised=data.standardised,
            geometry=data.geometry(root, where),
        )
        LOG.info("pretraining on %d sessions, %.2f GB on %s", len(sessions.runs), recording.gigabytes, where)
        return cls(config, device, Pool(recording, data.samples))

    def net(self, generator: torch.Generator) -> SentenceNet:
        """A fresh model whose scaler and whitener are fit on random windows."""
        sample = self.pool.sample(self.config.fit, generator).to(self.device)
        model = SentenceNet(Scaler(sample), context=self.config.context, channels=sample.shape[1]).to(self.device)
        model.whiten.fit(model.scale(sample))
        return model

    @staticmethod
    def spread(rows: Float[Tensor, "n h"]) -> Float[Tensor, " 2"]:
        """How varied unit-normalised `rows` are: their per-dimension std over `n` against an isotropic cloud's
        `h ** -0.5` (1 there, 0 when every row is the same), and the norm of their mean."""
        unit = functional.normalize(rows.float(), dim=-1)
        return torch.stack([unit.std(dim=0).mean() * math.sqrt(unit.shape[-1]), unit.mean(dim=0).norm()])

    def decay(self, step: int, steps: int) -> float:
        """The teacher's decay at `step` of `steps`."""
        config = self.config
        ramp = min(1.0, step / max(1.0, config.ema_ramp * steps))
        return config.ema_start + (config.ema - config.ema_start) * ramp

    def step(
        self, net: SentenceNet, predictor: Predictor, teacher: Target, generator: torch.Generator
    ) -> Float[Tensor, " 4"]:
        """One window batch's loss, the prediction's and target's spread, and the target's centre, not yet synced to
        the host; the gradients are left on the parameters."""
        config = self.config
        x = self.pool.sample(config.batch, generator).to(self.device)
        subject = torch.full((config.batch,), DEEP, dtype=torch.long, device=self.device)
        hidden = config.masking.draw(config.batch, x.shape[-1], config.data.rate, generator).to(self.device)
        with torch.autocast(self.device.type, dtype=torch.bfloat16):
            target = teacher.read(x, subject)
            shown = config.masking.apply(net.prepare(x, subject), hidden, predictor.token, generator)
            pred = predictor(teacher.convolutional(net).taps(shown)[-1])
        loss = LOSSES[config.loss](pred[hidden].float(), target[hidden])
        net.zero_grad(set_to_none=True)
        predictor.zero_grad(set_to_none=True)
        loss.backward()
        spread, (target_spread, centre) = self.spread(pred[hidden].detach()), self.spread(target[hidden])
        return torch.stack([loss.detach(), spread[0], target_spread, centre])

    def watch(self, stats: Float[Tensor, " 4"]) -> None:
        """Raise when `stats` (a mean over steps) show a non-finite loss or a spread under the floor."""
        loss, spread, target_spread, _ = stats.tolist()
        if not math.isfinite(loss):
            raise FloatingPointError("non-finite pretraining loss")
        floor = self.config.collapse_floor
        if min(spread, target_spread) < floor:
            raise CollapseError(f"spread {spread:.4f} (prediction), {target_spread:.4f} (target) under {floor}")

    def fit(self, net: SentenceNet, predictor: Predictor, generator: torch.Generator) -> tuple[Target, list[Round]]:
        """Train `net` and `predictor`; returns the teacher and each epoch's record."""
        config = self.config
        teacher = Target.of(net, config.top)
        per_epoch = math.ceil((config.windows or self.pool.total // self.pool.length) / config.batch)
        steps = per_epoch * config.epochs
        params = [*net.encoder(), *predictor.parameters()]
        optimizer = torch.optim.AdamW(
            params, lr=config.lr, weight_decay=config.weight_decay, fused=self.device.type == "cuda"
        )
        schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, config.lr, total_steps=steps)
        history: list[Round] = []
        for epoch in range(config.epochs):
            net.train()
            began = time.perf_counter()
            stats: list[Tensor] = []
            for step in range(per_epoch):
                stats.append(self.step(net, predictor, teacher, generator))
                nn.utils.clip_grad_norm_(params, config.clip)
                optimizer.step()
                schedule.step()
                teacher.follow(net, self.decay(epoch * per_epoch + step, steps))
                if (step + 1) % config.monitor == 0:
                    self.watch(torch.stack(stats[-config.monitor :]).mean(0))
            mean = torch.stack(stats).mean(0)
            self.watch(mean)
            loss, spread, target_spread, centre = mean.tolist()
            history.append(Round(epoch, loss, spread, target_spread, centre, time.perf_counter() - began))
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
