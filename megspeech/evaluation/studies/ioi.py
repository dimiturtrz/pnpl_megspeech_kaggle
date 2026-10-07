"""The inter-onset gaps a model is told, official test rows against Kaggle test rows, and what its score owes them.

`Timing` reads a split's span gaps (the values `Intervals.features` encodes) and rebuilds its onsets from other gaps:
each span's own gaps shuffled within it, gaps drawn from the Kaggle test's pool, or every gap that pool's median. The
model is scored on the replaced onsets with MEG and labels untouched (megspeech-7gf.105).

    uv run python -m megspeech.evaluation.studies.ioi sent40-ioi-ema9998-s0 --grouping kaggle
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


class Replacement(StrEnum):
    """Where a span's new gaps come from."""

    SHUFFLE = "shuffle"
    POOL = "pool"
    CONSTANT = "constant"

    def gaps(
        self,
        real: Float[Tensor, "g k"],
        pair: Bool[Tensor, "g k"],
        pool: Float[Tensor, " p"],
        generator: torch.Generator,
    ) -> Float[Tensor, "g k"]:
        """New gaps at every `pair` position of `real`, zero elsewhere."""
        drawn = {
            Replacement.SHUFFLE: self.shuffled,
            Replacement.POOL: self.sampled,
            Replacement.CONSTANT: self.median,
        }[self](real, pair, pool, generator)
        return drawn * pair

    @staticmethod
    def shuffled(
        real: Float[Tensor, "g k"], pair: Bool[Tensor, "g k"], pool: Float[Tensor, " p"], generator: torch.Generator
    ) -> Float[Tensor, "g k"]:
        keys = torch.rand(real.shape, generator=generator).to(real.device).where(pair, 2.0)
        return real.gather(1, keys.argsort(dim=1))

    @staticmethod
    def sampled(
        real: Float[Tensor, "g k"], pair: Bool[Tensor, "g k"], pool: Float[Tensor, " p"], generator: torch.Generator
    ) -> Float[Tensor, "g k"]:
        return pool[torch.randint(len(pool), real.shape, generator=generator).to(real.device)]

    @staticmethod
    def median(
        real: Float[Tensor, "g k"], pair: Bool[Tensor, "g k"], pool: Float[Tensor, " p"], generator: torch.Generator
    ) -> Float[Tensor, "g k"]:
        return pool.median().expand_as(real)


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

    @staticmethod
    def score(graded: Graded) -> str:
        """The macro with its bootstrap interval, then each domain's balanced accuracy and word count."""
        ruler = Ruler()
        interval = ruler.interval(ruler.macro(graded))
        domains = {name: f"{ruler.balanced(hits, y):.4f} n={len(y)}" for name, (hits, y) in graded.domains.items()}
        return f"macro {ruler.point(graded):.4f} [{interval.low:.4f}, {interval.high:.4f}] {domains}"

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

    @staticmethod
    def positioned(domains: dict[str, Split], seen: frozenset[tuple[int, ...]], graded: Graded) -> str:
        """`buckets` over the scored (unseen-span) words of every graded domain pooled."""
        placed = [Timing(domains[name]).places(domains[name].novel(seen)) for name in graded.domains]
        starts, ends = (torch.cat([place[k] for place in placed]) for k in range(2))
        hits = torch.cat([hit for hit, _ in graded.domains.values()])
        labels = torch.cat([label for _, label in graded.domains.values()])
        return Timing.buckets(starts, ends, hits, labels)

    def replaced(self, gaps: Float[Tensor, "g k"]) -> Split:
        """`split` whose onsets are rebuilt from `gaps` after each span's first word."""
        onset = torch.cat([gaps.new_zeros(len(gaps), 1), gaps.cumsum(dim=1)], dim=1)
        return replace(self.split, onset=onset)

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
    loaded = Loader.of(root, args.names, tree.parse(args), torch.device("cuda"))
    runs = Run.discover(root.raw())
    tau = loaded.report(runs).tau
    kaggle = Timing(loaded.kaggle(Track.DEEP))
    pool = kaggle.values()
    for partition in (Partition.VALIDATION, Partition.TEST):
        held = replace(loaded, partition=partition)
        domains, _ = held.domains(held.ruled(runs))
        if args.grid:
            LOG.info("kaggle-test %s", kaggle.grid())
            for name, split in domains.items():
                LOG.info("%s %s %s", partition, name, Timing(split).grid())
            continue
        if args.positions:
            graded = held.evaluator.graded(held.model, domains, held.log_prior, tau)
            report = Timing.positioned(domains, held.evaluator.seen, graded)
            LOG.info(
                "%s tau %s balanced top-10 (per-class recall over classes present in bucket): %s",
                partition,
                tau,
                report,
            )
            continue
        if args.blur:
            for label, blur in Timing.blurs(args.seed).items():
                swapped = {name: blur(Timing(split)) for name, split in domains.items()}
                graded = held.evaluator.graded(held.model, swapped, held.log_prior, tau)
                LOG.info("%s onsets=%s tau %s: %s", partition, label, tau, Timing.score(graded))
            continue
        LOG.info("kaggle-test %s", kaggle.summary())
        LOG.info("kaggle-test positions %s", kaggle.positions())
        for name, split in domains.items():
            LOG.info("%s %s %s", partition, name, Timing(split).summary())
            LOG.info("%s %s positions %s", partition, name, Timing(split).positions())
        for label in ("real", *Replacement):
            generator = torch.Generator().manual_seed(args.seed)
            swapped = {
                name: split
                if label == "real"
                else Timing(split).replaced(
                    Replacement(label).gaps(Timing(split).gaps(), Timing(split).pair(), pool, generator)
                )
                for name, split in domains.items()
            }
            graded = held.evaluator.graded(held.model, swapped, held.log_prior, tau)
            LOG.info("%s gaps=%s tau %s: %s", partition, label, tau, Timing.score(graded))


if __name__ == "__main__":
    main()
