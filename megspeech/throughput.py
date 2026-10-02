"""Where a training epoch's wall-clock goes: optimiser steps vs the per-epoch report, and the top ops of each.

`python -m megspeech.throughput --tasks Sherlock1` profiles a few steps on a small train set (the step cost
depends on the batch, not the train-set size) and one full validation report, then projects a full epoch.
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
from megspeech.train import Corpus, Step, TrainConfig, Trainer

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
class Projection:
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
        activities = [ProfilerActivity.CPU, *([ProfilerActivity.CUDA] if device.type == "cuda" else [])]
        sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
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
        sort = "self_cuda_time_total" if device.type == "cuda" and not self.host else "self_cpu_time_total"
        return Phase(seconds, host, on_device, events.table(sort_by=sort, row_limit=self.rows))

    def measure(self, trainer: Trainer, corpus: Corpus, full_spans: int) -> Projection:
        train = corpus.train
        model = trainer.model(train)
        optimizer = trainer.optimizer(model)
        total = self.warmup + 2 * self.steps
        state = Step(
            model, optimizer, torch.optim.lr_scheduler.OneCycleLR(optimizer, trainer.config.lr, total_steps=total)
        )
        draws = torch.randint(len(train.index), (total, trainer.config.batch), device=train.x.device)
        for spans in draws[: self.warmup]:
            trainer.step(state, train, spans)

        timed = iter(draws[self.warmup :].split(self.steps))

        def steps() -> None:
            for spans in next(timed):
                trainer.step(state, train, spans)

        log_prior = train.log_prior()
        stepped = self.phase(steps, train.x.device)
        reported = self.phase(lambda: trainer.report(model, corpus.validation, log_prior, 0, 0.0), train.x.device)
        per_epoch = math.ceil(full_spans / trainer.config.batch)
        return Projection(stepped, reported, stepped.seconds / self.steps, per_epoch)


def main() -> None:
    """Profile optimiser steps and one validation report; log top ops and the projected full-epoch split."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--tasks", nargs="*", default=["Sherlock1"])
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--full-spans", type=int, default=FULL_SPANS)
    parser.add_argument("--compiled", action="store_true", help="compile the model")
    parser.add_argument("--host", action="store_true", help="rank ops by host time")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    config = TrainConfig(tasks=tuple(args.tasks), compiled=args.compiled)
    root = DataRoot.from_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    corpus = Corpus.load(config, root, device)
    out = Throughput(steps=args.steps, host=args.host).measure(corpus.trainer(config, root), corpus, args.full_spans)
    for name, phase in (("train steps", out.train), ("report", out.report)):
        LOG.info(
            "%s: wall %.2fs host %.2fs device %.2fs\n%s", name, phase.seconds, phase.host, phase.device, phase.table
        )
    if device.type == "cuda":
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
