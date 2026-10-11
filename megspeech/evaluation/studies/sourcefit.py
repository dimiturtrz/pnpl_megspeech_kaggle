"""Training fit per source (megspeech-7gf.152): does pooling sources cost the subject-0 stem capacity?

A saved run is scored, at its own tau, on the fixed seeded subset of training spans that `Telemetry.fit` reads, with
the labelled words split by the source of the subject who heard them. Validation is the run's selection macro.

    uv run python -m megspeech.evaluation.studies.sourcefit --runs ema-bgmix05-w15-s0 ema-bgmix05-all-w15-s0
"""

import argparse
import logging
import math
from dataclasses import dataclass, replace
from enum import Enum, StrEnum

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from core.paths import DataRoot
from core.tree import Tree
from megspeech.data.batching.split import Split
from megspeech.data.corpus import Corpus
from megspeech.data.raw.runs import Run
from megspeech.device import Device
from megspeech.evaluation.group import Group
from megspeech.evaluation.loaded import LoadedModel
from megspeech.evaluation.loading import Loader, Scoring
from megspeech.evaluation.studies.gap import Condition, Locator
from megspeech.model.context.timing import KAGGLE_GAP

LOG = logging.getLogger(__name__)


class Source(Enum):
    """A recording source and the inclusive range of subject ids it owns."""

    LIBRIBRAIN = (0, 32)
    MASC = (101, 127)
    APPLESEED = (201, 212)
    KYMATA = (301, 323)

    def owns(self, subject: Int[Tensor, " n"]) -> Bool[Tensor, " n"]:
        low, high = self.value
        return (subject >= low) & (subject <= high)


UNSEEN_SUBJECT = 999
"""A subject id no sensor layer holds a matrix for: read through its unseen fallback."""


class Variant(StrEnum):
    """What the fit subset is read as: real, or with MEG windows swapped plus one more thing taken from the input."""

    REAL = "real"
    MEG = "meg"
    EVEN = "meg+even"
    GAPS = "meg+gaps"
    SUBJECT = "meg+subj"
    BOTH = "meg+even+subj"

    def apply(self, split: Split, seed: int) -> Split:
        """`split` as this variant reads it."""
        if self is Variant.REAL:
            return split
        shuffled = split.permuted(seed)
        timed = {
            Variant.MEG: lambda: shuffled,
            Variant.EVEN: lambda: Ablated(shuffled).evenly(),
            Variant.GAPS: lambda: Ablated(shuffled).regapped(seed),
            Variant.SUBJECT: lambda: Ablated(shuffled).anonymous(),
            Variant.BOTH: lambda: Ablated(Ablated(shuffled).evenly()).anonymous(),
        }
        return timed[self]()


@dataclass(frozen=True)
class Ablated:
    """A fit subset with one more thing taken from its input."""

    split: Split

    def evenly(self) -> Split:
        """Every span's words at the Kaggle test's typical spacing from zero: only the word count remains of timing."""
        split = self.split
        valid = split.index >= 0
        rank = (valid.cumsum(-1) - 1).to(split.onset.dtype)
        return replace(split, onset=torch.where(valid, rank * KAGGLE_GAP, split.onset))

    def regapped(self, seed: int) -> Split:
        """Every span's gaps between onsets dealt among its own gap slots in a random order."""
        split = self.split
        valid = split.index[:, 1:] >= 0
        gaps = split.onset[:, 1:] - split.onset[:, :-1]
        keys = torch.rand(gaps.shape, generator=torch.Generator().manual_seed(seed)).to(gaps.device)
        order = torch.where(valid, keys, 2.0).argsort(dim=1)
        moved = torch.cumsum(torch.where(valid, gaps.gather(1, order), 0.0), dim=1)
        return replace(split, onset=torch.cat([split.onset[:, :1], torch.where(valid, moved, split.onset[:, 1:])], 1))

    def anonymous(self) -> Split:
        return replace(self.split, subject=torch.full_like(self.split.subject, UNSEEN_SUBJECT))


@dataclass(frozen=True)
class SourceFit:
    """One run's fit per source under each variant, the words behind each source, and its validation macro."""

    fit: dict[Variant, dict[Source, float]]
    words: dict[Source, int]
    val: float
    tau: float

    @classmethod
    def of(cls, locator: Locator) -> "SourceFit":
        """The fit subset's context-head balanced top-10 within each source's labelled words, per variant."""
        fitted = locator.fitted
        heard = fitted.heard()
        subject, label = fitted.subject[heard], fitted.y[heard]
        masks = {source: source.owns(subject) for source in Source}
        fit = {
            variant: cls.scored(locator, locator.read(variant.apply(fitted, locator.seed)), label, masks)
            for variant in Variant
        }
        words = {source: int(mask.sum().item()) for source, mask in masks.items()}
        return cls(fit, words, locator.val(Condition.REAL), locator.tau)

    @staticmethod
    def scored(
        locator: Locator, context: Float[Tensor, "k v"], label: Int[Tensor, " k"], masks: dict[Source, Tensor]
    ) -> dict[Source, float]:
        evaluator, log_prior = locator.loaded.evaluator, locator.loaded.log_prior
        return {
            source: evaluator.evaluate(context[mask], label[mask], log_prior).by_tau[locator.tau]
            if mask.any()
            else math.nan
            for source, mask in masks.items()
        }

    def lines(self) -> list[str]:
        head = f"{'':12}" + "".join(f"{variant.value:>15}" for variant in Variant) + "  words"
        rows = [
            f"{source.name.lower():12}"
            + "".join(f"{self.fit[variant][source]:15.4f}" for variant in Variant)
            + f"  {self.words[source]}"
            for source in Source
        ]
        return [head, *rows, f"{'val macro':12}{self.val:.4f}  tau {self.tau}"]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=Device.CPU)
    tree = Tree(Scoring)
    tree.arguments(parser)
    args = parser.parse_args()
    root, device = DataRoot.from_config(), torch.device(args.device)
    scoring, runs = tree.parse(args), Run.discover(root.raw())
    for name in args.runs:
        loaded: LoadedModel = Loader.of(Group(root, scoring, device), [name])
        train = Corpus.load(loaded.data, root, device).train
        LOG.info("%s", name)
        for line in SourceFit.of(Locator.of(loaded, train, runs, args.seed)).lines():
            LOG.info("%s", line)


if __name__ == "__main__":
    main()
