"""Where the fit-versus-validation gap lives: text memorisation or MEG memorisation (megspeech-7gf.128).

A trained model is scored, at its own tau, on a fixed seeded subset of training spans (`fit`, as `Telemetry.fit` reads
it) and on validation (the selection macro over the unseen-span words of the counted domains), each under three
conditions. `real` reads the data as it is; `meg-shuffled` gives every word another word's MEG window, onsets kept
(`Split.permuted`); `order-shuffled` deals each span's words among its own slots, every word keeping its window and
label while the slots keep their onsets (`Split.ordered`). If the order shuffle costs fit far more than validation,
the model leans on training sentence text; if the MEG shuffle does, on training MEG.

    uv run python -m megspeech.evaluation.studies.gap sent40-ioi-ema9998-s0 --grouping kaggle --seed 0
"""

import argparse
import logging
from dataclasses import dataclass
from enum import StrEnum

import torch
from jaxtyping import Float
from torch import Tensor

from core.paths import DataRoot
from core.tree import Tree
from megspeech.data.batching.split import Split
from megspeech.data.corpus import Corpus
from megspeech.data.raw.runs import Run
from megspeech.evaluation.loaded import LoadedModel
from megspeech.evaluation.loading import Loader, Scoring
from megspeech.evaluation.scoring.score import Head
from megspeech.evaluation.scoring.telemetry import Telemetry

LOG = logging.getLogger(__name__)


class Condition(StrEnum):
    REAL = "real"
    MEG = "meg-shuffled"
    ORDER = "order-shuffled"

    def apply(self, split: Split, seed: int) -> Split:
        """`split` as this condition reads it."""
        return {
            Condition.REAL: lambda: split,
            Condition.MEG: lambda: split.permuted(seed),
            Condition.ORDER: lambda: split.ordered(seed),
        }[self]()


class Subset(StrEnum):
    FIT = "fit"
    VAL = "val"


@dataclass(frozen=True)
class Grid:
    """Balanced accuracy at top-10 per subset and condition."""

    scores: dict[Subset, dict[Condition, float]]

    def drop(self, subset: Subset, condition: Condition) -> float:
        """What `condition` costs `subset` against its real score."""
        return self.scores[subset][Condition.REAL] - self.scores[subset][condition]

    def gap(self, condition: Condition) -> float:
        """Fit less validation under `condition`."""
        return self.scores[Subset.FIT][condition] - self.scores[Subset.VAL][condition]

    def text_excess(self) -> float:
        """How much more the order shuffle costs fit than validation: the text-memorisation signature."""
        return self.drop(Subset.FIT, Condition.ORDER) - self.drop(Subset.VAL, Condition.ORDER)

    def meg_excess(self) -> float:
        """How much more the MEG shuffle costs fit than validation: the MEG-memorisation signature."""
        return self.drop(Subset.FIT, Condition.MEG) - self.drop(Subset.VAL, Condition.MEG)

    def lines(self) -> list[str]:
        head = "".join(f"{condition.value:>16}" for condition in Condition)
        rows = [f"{'':6}{head}"]
        rows += [
            f"{subset.value:6}" + "".join(f"{self.scores[subset][condition]:16.4f}" for condition in Condition)
            for subset in Subset
        ]
        rows.append(f"{'gap':6}" + "".join(f"{self.gap(condition):16.4f}" for condition in Condition))
        rows += [
            f"{subset.value} drop vs real: "
            + " ".join(f"{condition.value} {self.drop(subset, condition):.4f}" for condition in list(Condition)[1:])
            for subset in Subset
        ]
        rows.append(f"fit-minus-val drop: order {self.text_excess():.4f} meg {self.meg_excess():.4f}")
        return rows


@dataclass(frozen=True)
class Locator:
    """A loaded model's fit subset and validation domains, scored at `tau` under each condition."""

    loaded: LoadedModel
    fitted: Split
    domains: dict[str, Split]
    tau: float
    seed: int

    @classmethod
    def of(cls, loaded: LoadedModel, train: Split, runs: list[Run], seed: int) -> "Locator":
        """The fit subset sized as training telemetry sizes it, and the validation domains of `runs`."""
        domains, _ = loaded.domains(loaded.ruled(runs))
        words = sum(len(split.labelled()) for split in domains.values())
        return cls(loaded, Telemetry.subset(train, words, seed), domains, loaded.report(runs).tau, seed)

    def context(self, condition: Condition) -> Float[Tensor, "k v"]:
        """The context head's log-probabilities for the fit subset's labelled words."""
        return self.read(condition.apply(self.fitted, self.seed))

    def read(self, split: Split) -> Float[Tensor, "k v"]:
        """The context head's log-probabilities for the labelled words of `split`, a variant of the fit subset."""
        return self.loaded.context(split)[self.fitted.heard()]

    def fit(self, condition: Condition) -> float:
        """The context head's balanced top-10 on the fit subset's labelled words."""
        words = self.fitted.heard()
        return self.loaded.score(self.context(condition), self.fitted.y[words]).by_tau[self.tau]

    def val(self, condition: Condition) -> float:
        """The selection macro: the counted domains' unseen-span words, graded on the unshuffled split so every
        condition scores the same words."""
        evaluator = self.loaded.evaluator
        counted = {
            name: split
            for name, split in self.domains.items()
            if len(split.novel(evaluator.seen)) >= evaluator.min_novel
        }
        context = {
            name: evaluator.logits(self.loaded.model, condition.apply(split, self.seed))[Head.CONTEXT]
            for name, split in counted.items()
        }
        graded = evaluator.grade(context, counted, self.loaded.log_prior, self.tau)
        return evaluator.ruler.point(graded)

    def grid(self) -> Grid:
        return Grid(
            {
                Subset.FIT: {condition: self.fit(condition) for condition in Condition},
                Subset.VAL: {condition: self.val(condition) for condition in Condition},
            }
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="+")
    parser.add_argument("--seed", type=int, default=0)
    tree = Tree(Scoring)
    tree.arguments(parser)
    args = parser.parse_args()
    root, device = DataRoot.from_config(), torch.device("cuda")
    scoring, runs = tree.parse(args), Run.discover(root.raw())
    for name in args.names:
        loaded = Loader.of(root, [name], scoring, device)
        train = Corpus.load(loaded.data, root, device).train
        locator = Locator.of(loaded, train, runs, args.seed)
        LOG.info("%s tau %s seed %d", name, locator.tau, args.seed)
        for line in locator.grid().lines():
            LOG.info("%s", line)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
