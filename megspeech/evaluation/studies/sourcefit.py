"""Training fit per source (megspeech-7gf.152): does pooling sources cost the subject-0 stem capacity?

A saved run is scored, at its own tau, on the fixed seeded subset of training spans that `Telemetry.fit` reads, with
the labelled words split by the source of the subject who heard them. Validation is the run's selection macro.

    uv run python -m megspeech.evaluation.studies.sourcefit --runs ema-bgmix05-w15-s0 ema-bgmix05-all-w15-s0
"""

import argparse
import logging
import math
from dataclasses import dataclass
from enum import Enum

import torch

from core.paths import DataRoot
from core.tree import Tree
from megspeech.data.corpus import Corpus
from megspeech.data.raw.runs import Run
from megspeech.evaluation.loaded import LoadedModel
from megspeech.evaluation.loading import Loader, Scoring
from megspeech.evaluation.studies.gap import Condition, Locator

LOG = logging.getLogger(__name__)


class Source(Enum):
    """A recording source and the inclusive range of subject ids it owns."""

    LIBRIBRAIN = (0, 32)
    MASC = (101, 127)
    APPLESEED = (201, 212)
    KYMATA = (301, 323)

    def owns(self, subject: torch.Tensor) -> torch.Tensor:
        low, high = self.value
        return (subject >= low) & (subject <= high)


@dataclass(frozen=True)
class SourceFit:
    """One run's fit per source, the words behind each, and its validation macro."""

    fit: dict[Source, float]
    words: dict[Source, int]
    val: float
    tau: float

    @classmethod
    def of(cls, locator: Locator) -> "SourceFit":
        """The fit subset's context-head balanced top-10 within each source's labelled words."""
        evaluator, fitted = locator.loaded.evaluator, locator.fitted
        heard = fitted.heard()
        context = locator.context(Condition.REAL)
        subject, label = fitted.subject[heard], fitted.y[heard]
        fit: dict[Source, float] = {}
        words: dict[Source, int] = {}
        for source in Source:
            mask = source.owns(subject)
            words[source] = int(mask.sum().item())
            fit[source] = (
                evaluator.evaluate(context[mask], label[mask], locator.loaded.log_prior).by_tau[locator.tau]
                if words[source]
                else math.nan
            )
        return cls(fit, words, locator.val(Condition.REAL), locator.tau)

    def lines(self) -> list[str]:
        rows = [f"{source.name.lower():12}fit {self.fit[source]:.4f}  words {self.words[source]}" for source in Source]
        return [*rows, f"{'val macro':12}{self.val:.4f}  tau {self.tau}"]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    tree = Tree(Scoring)
    tree.arguments(parser)
    args = parser.parse_args()
    root, device = DataRoot.from_config(), torch.device(args.device)
    scoring, runs = tree.parse(args), Run.discover(root.raw())
    for name in args.runs:
        loaded: LoadedModel = Loader.of(root, [name], scoring, device)
        train = Corpus.load(loaded.data, root, device).train
        LOG.info("%s", name)
        for line in SourceFit.of(Locator.of(loaded, train, runs, args.seed)).lines():
            LOG.info("%s", line)


if __name__ == "__main__":
    main()
