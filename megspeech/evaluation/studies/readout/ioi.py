"""The inter-onset gaps a model is told, official test rows against Kaggle test rows, and what its score owes them.

`Timing` reads a split's span gaps (the values `Intervals.features` encodes) and rebuilds its onsets from other gaps:
each span's own gaps shuffled within it, gaps drawn from the Kaggle test's pool, or every gap that pool's median. The
model is scored on the replaced onsets with MEG and labels untouched (megspeech-7gf.105).

    uv run python -m megspeech.evaluation.studies.readout.ioi sent40-ioi-ema9998-s0 --grouping kaggle
"""

import argparse
import logging
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import StrEnum

import numpy as np
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from core.paths import DataRoot
from core.tree import Tree
from megspeech.data.batching.split import Split
from megspeech.data.cache.kaggle import Track
from megspeech.data.raw.runs import Partition, Run
from megspeech.data.spans.spans import MAX_WORDS, PAD
from megspeech.device import Device
from megspeech.evaluation.group import Group
from megspeech.evaluation.loaded import LoadedModel
from megspeech.evaluation.loading import Loader
from megspeech.evaluation.metric.ruler import Graded, Ruler
from megspeech.evaluation.scoring.config import Scoring
from megspeech.model.context.timing import Jitter
from megspeech.model.input.subjects import Fallback, Unseen

LOG = logging.getLogger(__name__)

QUANTILES = (0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)
GRID = 50.0
"""Hz of the cache: the share of gaps landing on its samples says whether an onset source was rounded to them."""
GRID_TOLERANCE = 1e-3
GRIDS = {"1/250": 250.0, "1/1000": 1000.0, "1/100": 100.0, "1/50": 50.0}
"""Sample rates whose grids a gap may sit on: the Kaggle archives' 250 Hz, millisecond, 10 ms and the cache's 50 Hz."""
GRID_EXACT = 1e-6
"""Seconds a gap may be off a grid point and still sit on it (float32 onsets)."""
GAP_QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)
JITTER_MS = (5, 10, 20, 40, 80)
QUANTISE_MS = (20, 50)
MS = 1000.0


@dataclass(frozen=True)
class Redraw:
    """A span table's `real` gaps, which of them exist (`pair`), the `pool` new gaps come from and the draw's
    `generator`."""

    real: Float[Tensor, "g k"]
    pair: Bool[Tensor, "g k"]
    pool: Float[Tensor, " p"]
    generator: torch.Generator

    def shuffled(self) -> Float[Tensor, "g k"]:
        keys = torch.rand(self.real.shape, generator=self.generator).to(self.real.device).where(self.pair, 2.0)
        return self.real.gather(1, keys.argsort(dim=1))

    def sampled(self) -> Float[Tensor, "g k"]:
        return self.pool[torch.randint(len(self.pool), self.real.shape, generator=self.generator).to(self.real.device)]

    def median(self) -> Float[Tensor, "g k"]:
        return self.pool.median().expand_as(self.real)


class Replacement(StrEnum):
    """Where a span's new gaps come from."""

    SHUFFLE = "shuffle"
    POOL = "pool"
    CONSTANT = "constant"

    def gaps(self, draw: Redraw) -> Float[Tensor, "g k"]:
        """New gaps at every `pair` position of the draw's real gaps, zero elsewhere."""
        drawn = {
            Replacement.SHUFFLE: draw.shuffled,
            Replacement.POOL: draw.sampled,
            Replacement.CONSTANT: draw.median,
        }[self]()
        return drawn * draw.pair


@dataclass(frozen=True)
class Timing:
    """The onset gaps inside `split`'s spans."""

    split: Split

    def pair(self) -> Bool[Tensor, "g k"]:
        """Which gaps of a span exist: both of its neighbours are words."""
        present = self.split.index != PAD
        return present[:, 1:] & present[:, :-1]

    def gaps(self) -> Float[Tensor, "g k"]:
        """Seconds between each word's onset and the next's, zero where there is no next word."""
        return (self.split.onset[:, 1:] - self.split.onset[:, :-1]) * self.pair()

    def values(self) -> Float[Tensor, " n"]:
        return self.gaps()[self.pair()]

    def summary(self) -> str:
        """Count, mean, quantiles of the gaps and the share of spans that are one word, as the model's gaps."""
        values = self.values().cpu().numpy()
        lengths = (self.split.index != PAD).sum(dim=1)
        cuts = np.quantile(values, QUANTILES)
        quantiles = " ".join(f"p{round(100 * q)}={v:.3f}" for q, v in zip(QUANTILES, cuts, strict=True))
        zeros = float((values <= 0).mean())
        alone = float((lengths == 1).float().mean())
        grid = float((np.abs(values * GRID - np.round(values * GRID)) < GRID_TOLERANCE).mean())
        head = f"gaps n={len(values)} mean={values.mean():.3f}"
        return f"{head} {quantiles} zero={zeros:.4f} alone={alone:.3f} on-50Hz={grid:.3f}"

    def grid(self) -> str:
        """The share of gaps on each sample grid, then the minimum and the quantiles of the gaps in milliseconds."""
        values = self.values().cpu().double().numpy()
        shares = " ".join(
            f"on-{name}={float((np.abs(values * rate - np.round(values * rate)) < GRID_EXACT * rate).mean()):.3f}"
            for name, rate in GRIDS.items()
        )
        cuts = np.quantile(values, GAP_QUANTILES) * MS
        quantiles = " ".join(f"q{round(100 * q)}={v:.1f}" for q, v in zip(GAP_QUANTILES, cuts, strict=True))
        return f"gaps n={len(values)} min={values.min() * MS:.1f}ms {quantiles} (ms) {shares}"

    def positions(self) -> str:
        """Median gap and count at each place in the span."""
        gaps, pair = self.gaps().cpu().numpy(), self.pair().cpu().numpy()
        places = [k for k in range(MAX_WORDS - 1) if pair[:, k].any()]
        return " ".join(f"{k}:{np.median(gaps[pair[:, k], k]):.3f}(n={pair[:, k].sum()})" for k in places)

    @staticmethod
    def blurs(seed: int) -> dict[str, Callable[["Timing"], Split]]:
        """Each onset-blurring arm by label: the real onsets, jitter at every sigma, and the quantised grids."""
        arms: dict[str, Callable[[Timing], Split]] = {"real": lambda timing: timing.split}
        for ms in JITTER_MS:
            arms[f"jitter{ms}ms"] = lambda timing, ms=ms: timing.jittered(ms / MS, torch.Generator().manual_seed(seed))
        for ms in QUANTISE_MS:
            arms[f"quant{ms}ms"] = lambda timing, ms=ms: timing.quantised(ms / MS)
        return arms

    def places(self, words: Int[Tensor, " n"]) -> tuple[Int[Tensor, " n"], Int[Tensor, " n"]]:
        """Each of `words` (ids into the split's word rows) by its place counted from its span's start and end."""
        index = self.split.index
        present = index != PAD
        start = torch.arange(index.shape[1], device=index.device).expand_as(index)
        end = present.sum(dim=1, keepdim=True) - 1 - start
        from_start, from_end = (torch.zeros(len(self.split.y), dtype=torch.long, device=index.device) for _ in range(2))
        from_start[index[present]] = start[present]
        from_end[index[present]] = end[present]
        return from_start[words], from_end[words]

    @staticmethod
    def buckets(
        start: Int[Tensor, " n"], end: Int[Tensor, " n"], hits: Bool[Tensor, " n"], labels: Int[Tensor, " n"]
    ) -> str:
        """Top-k balanced accuracy (per-class recall averaged over the classes present in the bucket) and word count
        of each place from the span's end, then from its start; the last bucket of each pools every later place."""
        ruler = Ruler()
        arms = {
            "from-end": (end, ("last", "second-last", "third-last", "earlier")),
            "from-start": (start, ("first", "second", "rest")),
        }
        parts = []
        for arm, (place, names) in arms.items():
            capped = place.clamp(max=len(names) - 1)
            cells = []
            for k, name in enumerate(names):
                mask = capped == k
                accuracy = ruler.balanced(hits[mask], labels[mask]) if mask.any() else float("nan")
                cells.append(f"{name}={accuracy:.4f} n={int(mask.sum())}")
            parts.append(f"{arm} " + " ".join(cells))
        return " | ".join(parts)

    def replaced(self, gaps: Float[Tensor, "g k"]) -> Split:
        """`split` whose onsets are rebuilt from `gaps` after each span's first word."""
        onset = torch.cat([gaps.new_zeros(len(gaps), 1), gaps.cumsum(dim=1)], dim=1)
        return replace(self.split, onset=onset)

    def redrawn(self, replacement: Replacement, pool: Float[Tensor, " p"], generator: torch.Generator) -> Split:
        """`split` whose gaps `replacement` drew anew from `pool`."""
        return self.replaced(replacement.gaps(Redraw(self.gaps(), self.pair(), pool, generator)))

    def blurred(self, onset: Float[Tensor, "g w"]) -> Split:
        """`split` at `onset`, kept in order among its words and re-zeroed at each span's first word."""
        return replace(self.split, onset=Jitter.ordered(onset, self.split.index != PAD, self.split.onset))

    def jittered(self, sigma: float, generator: torch.Generator) -> Split:
        """Every word onset moved by Gaussian noise of `sigma` seconds."""
        noise = torch.randn(self.split.onset.shape, generator=generator).to(self.split.onset.device)
        return self.blurred(self.split.onset + sigma * noise)

    def quantised(self, step: float) -> Split:
        """Every word onset rounded to the grid of `step` seconds."""
        return self.blurred(torch.round(self.split.onset / step) * step)


@dataclass(frozen=True)
class Scored:
    """A model's `graded` hits over some domains, as the study reports them."""

    graded: Graded

    def score(self) -> str:
        """The macro with its bootstrap interval, then each domain's balanced accuracy and word count."""
        ruler, graded = Ruler(), self.graded
        interval = ruler.interval(ruler.macro(graded))
        domains = {name: f"{ruler.balanced(hits, y):.4f} n={len(y)}" for name, (hits, y) in graded.domains.items()}
        return f"macro {ruler.point(graded):.4f} [{interval.low:.4f}, {interval.high:.4f}] {domains}"

    def positioned(self, domains: dict[str, Split], seen: frozenset[tuple[int, ...]]) -> str:
        """`Timing.buckets` over the scored (unseen-span) words of every graded domain pooled."""
        graded = self.graded.domains
        placed = [Timing(domains[name]).places(domains[name].novel(seen)) for name in graded]
        starts, ends = (torch.cat([place[k] for place in placed]) for k in range(2))
        hits = torch.cat([hit for hit, _ in graded.values()])
        labels = torch.cat([label for _, label in graded.values()])
        return Timing.buckets(starts, ends, hits, labels)


class Mode(StrEnum):
    """What the study reports per partition: the flag that asks for it, the gap replacements when none does."""

    GRID = "grid"
    POSITIONS = "positions"
    BLUR = "blur"
    GAPS = "gaps"

    @classmethod
    def chosen(cls, args: argparse.Namespace) -> "Mode":
        """The first mode whose flag `args` set, in the order the flags take precedence."""
        return next((mode for mode in (cls.GRID, cls.POSITIONS, cls.BLUR) if getattr(args, mode)), cls.GAPS)


@dataclass(frozen=True)
class Held:
    """One partition's `domains` as `loaded` scores them at `tau`, beside the Kaggle test's timing."""

    loaded: LoadedModel
    domains: dict[str, Split]
    tau: float
    kaggle: Timing
    seed: int

    def graded(self, domains: dict[str, Split]) -> Scored:
        loaded = self.loaded
        return Scored(loaded.evaluator.graded(loaded.model, domains, loaded.calibration(self.tau)))

    def report(self, mode: Mode) -> None:
        {Mode.GRID: self.grid, Mode.POSITIONS: self.positions, Mode.BLUR: self.blur, Mode.GAPS: self.gaps}[mode]()

    def grid(self) -> None:
        """The sample grids of the Kaggle test's gaps and each domain's."""
        LOG.info("kaggle-test %s", self.kaggle.grid())
        for name, split in self.domains.items():
            LOG.info("%s %s %s", self.loaded.partition, name, Timing(split).grid())

    def positions(self) -> None:
        """The score of the words by their place in the span."""
        report = self.graded(self.domains).positioned(self.domains, self.loaded.evaluator.seen)
        LOG.info(
            "%s tau %s balanced top-10 (per-class recall over classes present in bucket): %s",
            self.loaded.partition,
            self.tau,
            report,
        )

    def blur(self) -> None:
        """The score under every onset-blurring arm."""
        for label, blur in Timing.blurs(self.seed).items():
            scored = self.graded({name: blur(Timing(split)) for name, split in self.domains.items()})
            LOG.info("%s onsets=%s tau %s: %s", self.loaded.partition, label, self.tau, scored.score())

    def gaps(self) -> None:
        """The gap statistics of the Kaggle test and each domain, then the score under real and replaced gaps."""
        partition = self.loaded.partition
        LOG.info("kaggle-test %s", self.kaggle.summary())
        LOG.info("kaggle-test positions %s", self.kaggle.positions())
        for name, split in self.domains.items():
            LOG.info("%s %s %s", partition, name, Timing(split).summary())
            LOG.info("%s %s positions %s", partition, name, Timing(split).positions())
        LOG.info("%s gaps=%s tau %s: %s", partition, "real", self.tau, self.graded(self.domains).score())
        pool = self.kaggle.values()
        for replacement in Replacement:
            generator = torch.Generator().manual_seed(self.seed)
            swapped = {
                name: Timing(split).redrawn(replacement, pool, generator) for name, split in self.domains.items()
            }
            LOG.info("%s gaps=%s tau %s: %s", partition, replacement, self.tau, self.graded(swapped).score())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="+")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grid", action="store_true", help="report the sample grid and gap quantiles instead")
    parser.add_argument("--positions", action="store_true", help="score words by place in their span instead")
    parser.add_argument("--blur", action="store_true", help="score jittered and quantised onsets instead")
    tree = Tree(Scoring, Scoring(fallback=Fallback(Unseen.DEEP)))
    tree.arguments(parser)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    root = DataRoot.from_config()
    loaded = Loader.of(Group(root, tree.parse(args), Device.CUDA.as_torch()), args.names)
    runs = Run.discover(root.raw())
    tau = loaded.report(runs).tau
    kaggle = Timing(loaded.kaggle(Track.DEEP))
    mode = Mode.chosen(args)
    for partition in (Partition.VALIDATION, Partition.TEST):
        held = replace(loaded, partition=partition)
        domains, _ = held.domains(held.ruled(runs))
        Held(held, domains, tau, kaggle, args.seed).report(mode)


if __name__ == "__main__":
    main()
