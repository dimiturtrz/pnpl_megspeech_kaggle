"""Where a training epoch's wall-clock goes: optimiser steps vs the per-epoch report, and the top ops of each.

`python -m megspeech.training.cli.throughput --data.tasks Sherlock1` profiles a few steps on a small train set (the
step cost depends on the batch, not the train-set size) and one full validation report, then projects a full epoch.
"""

import argparse
import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch.profiler import ProfilerActivity, profile

from core.paths import DataRoot
from core.tree import Tree
from megspeech.data.config import DataConfig
from megspeech.data.corpus import Corpus
from megspeech.data.raw.runs import SHERLOCK
from megspeech.device import Device
from megspeech.training.config import RunConfig
from megspeech.training.loop.trainer import Trainer

LOG = logging.getLogger(__name__)

STEPS = 50
WARMUP = 5
ROWS = 20
FULL_SPANS = 117486
"""Spans in the full default train set (logged by every training run)."""


@dataclass(frozen=True)
class Phase:
    """Wall seconds of one profiled phase and its top-op table."""

    seconds: float
    host: float
    device: float
    table: str


@dataclass(frozen=True)
class EpochEstimate:
    train: Phase
    report: Phase
    per_step: float
    steps: int

    def epoch(self) -> float:
        """Projected seconds of one full epoch: every optimiser step plus the report."""
        return self.per_step * self.steps + self.report.seconds

    def eval_share(self) -> float:
        return self.report.seconds / self.epoch()


@dataclass(frozen=True)
class Throughput:
    steps: int = STEPS
    warmup: int = WARMUP
    rows: int = ROWS
    host: bool = False

    def phase(self, work: Callable[[], object], device: torch.device) -> Phase:
        """Time `work` unprofiled and synchronised, then run it again under the profiler for the op table."""
        activities = [ProfilerActivity.CPU, *([ProfilerActivity.CUDA] if device.type == Device.CUDA else [])]
        sync = torch.cuda.synchronize if device.type == Device.CUDA else (lambda: None)
        start = time.perf_counter()
        work()
        sync()
        seconds = time.perf_counter() - start
        with profile(activities=activities) as prof:
            work()
            sync()
        events = prof.key_averages()
        host = sum(e.self_cpu_time_total for e in events) / 1e6
        on_device = sum(e.self_device_time_total for e in events) / 1e6
        sort = "self_cuda_time_total" if device.type == Device.CUDA and not self.host else "self_cpu_time_total"
        return Phase(seconds, host, on_device, events.table(sort_by=sort, row_limit=self.rows))

    def measure(self, trainer: Trainer, corpus: Corpus, full_spans: int) -> EpochEstimate:
        train = corpus.train
        model = trainer.model(train)
        total = self.warmup + 2 * self.steps
        state = trainer.state(model, train, total)
        draws = torch.randint(len(train.index), (total, trainer.config.optim.batch), device=train.y.device)
        for spans in draws[: self.warmup]:
            trainer.step(state, train, spans)

        timed = iter(draws[self.warmup :].split(self.steps))

        def steps() -> None:
            for spans in next(timed):
                trainer.step(state, train, spans)

        log_prior = train.log_prior()
        evaluator = trainer.evaluator
        stepped = self.phase(steps, train.y.device)
        reported = self.phase(lambda: evaluator.report(model, corpus.validation, log_prior, 0, 0.0), train.y.device)
        per_epoch = math.ceil(full_spans / trainer.config.optim.batch)
        return EpochEstimate(stepped, reported, stepped.seconds / self.steps, per_epoch)


def main() -> None:
    """Profile optimiser steps and one validation report; log top ops and the projected full-epoch split."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--full-spans", type=int, default=FULL_SPANS)
    parser.add_argument("--host", action="store_true", help="rank ops by host time")
    tree = Tree(RunConfig, RunConfig(data=DataConfig(tasks=(SHERLOCK,))))
    tree.arguments(parser)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    config = tree.parse(args)
    root = DataRoot.from_config()
    device = Device.best().as_torch()
    corpus = Corpus.load(config.data, root, device)
    out = Throughput(steps=args.steps, host=args.host).measure(
        Trainer.of(corpus, config, root), corpus, args.full_spans
    )
    for name, phase in (("train steps", out.train), ("report", out.report)):
        LOG.info(
            "%s: wall %.2fs host %.2fs device %.2fs\n%s", name, phase.seconds, phase.host, phase.device, phase.table
        )
    if device.type == Device.CUDA:
        LOG.info("peak reserved %.1f GB", torch.cuda.max_memory_reserved() / 1e9)
    LOG.info(
        "per step %.1f ms x %d steps + report %.1fs = epoch %.1fs (%.2f epochs/min), eval share %.0f%%",
        1000 * out.per_step,
        out.steps,
        out.report.seconds,
        out.epoch(),
        60 / out.epoch(),
        100 * out.eval_share(),
    )


if __name__ == "__main__":
    main()
