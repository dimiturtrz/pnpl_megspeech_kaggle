"""Cross-subject pooling on the ruler's Sherlock1 text: subject 0's prediction for a word combined with those of the
other subjects who heard the same word (`Text` keys the match).

Each subject's context-head log-softmax is averaged with a weight: `w0` for subject 0, the `Weighting`'s for the
others, so `w0 = inf` is subject 0 alone. A subject's reliability is its balanced top-10 above chance on its own
Sherlock1 session-12 run, the text of subject 0's test run, which a subject-0 model never trained on (a model trained
on other subjects' runs trained on it, so its reliabilities are optimistic).

    uv run python -m megspeech.evaluation.studies pool.subjects sent40-s7 [--grouping ...]
"""

import argparse
import logging
import math
from dataclasses import dataclass
from typing import Protocol

import pandas as pd
import torch
from jaxtyping import Bool, Float
from torch import Tensor

from core.paths import DataRoot
from core.tree import Tree
from megspeech.data.labels.vocabulary import VOCAB_SIZE
from megspeech.data.raw.runs import Run
from megspeech.device import Device
from megspeech.evaluation.loaded.ensemble import Loader
from megspeech.evaluation.loaded.group import Group
from megspeech.evaluation.loaded.model import LoadedModel
from megspeech.evaluation.metric.ruler import TOP
from megspeech.evaluation.scoring.config import Scoring
from megspeech.evaluation.scoring.score import Score
from megspeech.evaluation.text import RULER, Text

LOG = logging.getLogger(__name__)

CHANCE = TOP / VOCAB_SIZE
"""Balanced top-10 of a predictor that ignores the MEG."""

RELIABILITY_RUN = (RULER, 12, 1)
"""A pooled subject's run of subject 0's test text, the one its reliability is read on."""

WEIGHTS = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, math.inf)
"""Subject 0's weights tried against each other subject's, doubling past the 32 others' total; `inf` is subject 0
alone."""


class Weighting(Protocol):
    """Each other subject's weight from its reliability."""

    def weights(self, reliability: Float[Tensor, " s"]) -> Float[Tensor, " s"]: ...


class Equal:
    """Every other subject weighs 1."""

    def weights(self, reliability: Float[Tensor, " s"]) -> Float[Tensor, " s"]:
        return torch.ones_like(reliability)


class Reliable:
    """Each other subject weighs its reliability above chance, scaled to a mean of 1; one at chance weighs 0."""

    def weights(self, reliability: Float[Tensor, " s"]) -> Float[Tensor, " s"]:
        above = (reliability - CHANCE).clamp(min=0)
        return above / above.mean().clamp(min=torch.finfo(above.dtype).tiny)


WEIGHTINGS: dict[str, Weighting] = {"equal": Equal(), "reliable": Reliable()}


@dataclass(frozen=True)
class AlignedSubjects:
    """The other subjects' context logits aligned to subject 0's ruler words, which of them heard each word, and
    each subject's reliability."""

    logits: Float[Tensor, "s n v"]
    present: Bool[Tensor, "s n"]
    reliability: Float[Tensor, " s"]

    @property
    def heard(self) -> Bool[Tensor, " n"]:
        """Which ruler words some other subject heard."""
        return self.present.any(dim=0)

    @classmethod
    def of(cls, loaded: LoadedModel, reference: pd.DataFrame, runs: list[Run]) -> "AlignedSubjects":
        """Every pooled subject's `aligned` logits, in subject order."""
        pooled = Run.pooled(runs)
        trained = {run.subject: run for run in runs if run.key == RELIABILITY_RUN}
        aligned = [cls.aligned(loaded, reference, run, trained[run.subject]) for run in pooled]
        reliability = torch.tensor([bacc for *_, bacc in aligned], device=loaded.device)
        LOG.info(
            "reliability %s", {run.subject: round(float(r), 4) for run, r in zip(pooled, reliability, strict=True)}
        )
        return cls(torch.stack([a[0] for a in aligned]), torch.stack([a[1] for a in aligned]), reliability)

    @staticmethod
    def aligned(
        loaded: LoadedModel, reference: pd.DataFrame, pooled: Run, trained: Run
    ) -> tuple[Float[Tensor, "n v"], Bool[Tensor, " n"], float]:
        """One other subject's context logits at each of `reference`'s words (0 where it did not hear the word),
        which words it heard, and its reliability on its `trained` run."""
        splits = loaded.splits([pooled, trained])
        rows, split = splits[pooled]
        at = torch.from_numpy(Text(reference).matches(Text(rows))).to(split.y.device)
        aligned = torch.where((at >= 0)[:, None], loaded.context(split)[at.clamp(min=0)], 0.0)
        return aligned, at >= 0, loaded.labelled(splits[trained][1]).bacc10


@dataclass(frozen=True)
class Pool:
    """Subject 0's log-softmax weighted `w0` against the others', each weighted by `weighting`."""

    w0: float
    weighting: Weighting

    def combine(self, reference: Float[Tensor, "n v"], aligned: AlignedSubjects) -> Float[Tensor, "n v"]:
        """The weighted mean of log-softmax over subject 0 and the other subjects present at each word."""
        relative = self.weighting.weights(aligned.reliability)[:, None] / self.w0 * aligned.present
        total = reference.log_softmax(-1) + (relative[..., None] * aligned.logits.log_softmax(-1)).sum(0)
        return total / (1 + relative.sum(0))[:, None]


def main() -> None:
    """Log the ruler domain and the macro with subject 0 alone and pooled, for every weighting and `w0`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="+", help="trained models under models/, averaged as one")
    tree = Tree(Scoring)
    tree.arguments(parser)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    root = DataRoot.from_config()
    loaded = Loader.of(Group(root, tree.parse(args), Device.CUDA.as_torch()), args.names)
    evaluator, prior, runs = loaded.evaluator, loaded.log_prior, Run.discover(root.raw())
    domains, reference = loaded.domains(loaded.data.held(runs))
    context = {name: loaded.context(split) for name, split in domains.items()}
    aligned = AlignedSubjects.of(loaded, reference, runs)
    LOG.info("ruler words %d, heard by another subject %d", len(aligned.heard), int(aligned.heard.sum()))
    novel = domains[RULER].novel(evaluator.seen)
    shared = novel[aligned.heard[novel]]
    for label, weighting in WEIGHTINGS.items():
        for w0 in WEIGHTS:
            pooled = Pool(w0, weighting).combine(context[RULER], aligned)
            matched = evaluator.evaluate(pooled[shared], domains[RULER].y[shared], prior)
            scores = {
                name: evaluator.unseen(logits, domains[name], prior)
                for name, logits in (context | {RULER: pooled}).items()
            }
            ruler, macro = scores[RULER], Score.mean(evaluator.counted(scores).values())
            LOG.info(
                "%-8s w0 %-4g matched %.4f (n %d)  ruler %.4f (n %d)  macro %.4f (n %d) tau %g",
                label,
                w0,
                matched.bacc10,
                matched.n,
                ruler.bacc10,
                ruler.n,
                macro.bacc10,
                macro.n,
                macro.tau,
            )


if __name__ == "__main__":
    main()
