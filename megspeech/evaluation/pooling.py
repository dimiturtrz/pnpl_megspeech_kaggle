"""Cross-subject pooling on the ruler's Sherlock1 text: subject 0's prediction for a word combined with those of the
other subjects who heard the same word (`megspeech.evaluation.overlap` keys the match).

Each subject's context-head log-softmax is averaged with a weight: `w0` for subject 0, the `Weighting`'s for the
others, so `w0 = inf` is subject 0 alone. A subject's reliability is its balanced top-10 above chance on its own
Sherlock1 session-12 run, the text of subject 0's test run, which a subject-0 model never trained on (a model trained
on other subjects' runs trained on it, so its reliabilities are optimistic).

    uv run python -m megspeech.evaluation.pooling sent40-s7
"""

import argparse
import logging
import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
import torch
from jaxtyping import Bool, Float
from torch import Tensor, nn

from core.paths import DataRoot
from megspeech.data.continuous import Continuous
from megspeech.data.corpus import Corpus, DataConfig
from megspeech.data.grouping import Grouping
from megspeech.data.runs import Partition, Run
from megspeech.data.split import Split
from megspeech.data.vocabulary import WORDS
from megspeech.evaluation.overlap import RULER, Text
from megspeech.evaluation.report import Evaluator, Head, Score
from megspeech.evaluation.ruler import TOP
from megspeech.model.ensemble import Ensemble
from megspeech.model.subjects import Unseen

LOG = logging.getLogger(__name__)

CHANCE = TOP / WORDS
"""Balanced top-10 of a predictor that ignores the MEG."""

TRAINED = ("Sherlock1", 12, 1)
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
class Pool:
    """Subject 0's log-softmax weighted `w0` against the others', each weighted by `weighting`."""

    w0: float
    weighting: Weighting

    def combine(
        self,
        reference: Float[Tensor, "n v"],
        others: Float[Tensor, "s n v"],
        present: Bool[Tensor, "s n"],
        reliability: Float[Tensor, " s"],
    ) -> Float[Tensor, "n v"]:
        """The weighted mean of log-softmax over subject 0 and the other subjects `present` at each word."""
        relative = self.weighting.weights(reliability)[:, None] / self.w0 * present
        total = reference.log_softmax(-1) + (relative[..., None] * others.log_softmax(-1)).sum(0)
        return total / (1 + relative.sum(0))[:, None]


@dataclass(frozen=True)
class Echoes:
    """The other subjects' context logits aligned to subject 0's ruler words, which of them heard each word, and
    each subject's reliability."""

    logits: Float[Tensor, "s n v"]
    present: Bool[Tensor, "s n"]
    reliability: Float[Tensor, " s"]


@dataclass(frozen=True)
class Pooling:
    """One trained model scored on validation with the ruler's text pooled across subjects."""

    root: DataRoot
    model: nn.Module
    data: DataConfig
    evaluator: Evaluator
    log_prior: Float[Tensor, " v"]

    @classmethod
    def of(
        cls, root: DataRoot, name: str, grouping: Grouping, device: torch.device, unseen: Unseen = Unseen.IDENTITY
    ) -> "Pooling":
        """The model `name`, reading a subject its subject layer has no matrix for through `unseen`."""
        directory = root.models() / name
        model = Ensemble.load([directory / "model.pt"], unseen).to(device)
        data = DataConfig.shared([directory], grouping)
        heard = Corpus.heard(data.decimator(root).cache, Run.discover(root.raw()), data)
        labels = heard["target"].to_numpy()
        log_prior = Split.prior(torch.tensor(labels), torch.ones(len(labels))).to(device)
        return cls(root, model, data, Evaluator(device, Corpus.sequences(heard, grouping)), log_prior)

    def load(self, runs: list[Run]) -> tuple[Continuous, pd.DataFrame]:
        """`runs` on the device, and their words keyed by `Text` and cut as the model's grouping cuts them."""
        cache = self.data.decimator(self.root).cache
        names = [r.name for r in runs]
        device = self.evaluator.device
        source = Continuous.load(cache, device, names, standardised=self.data.standardised, rate=self.data.rate)
        keyed = Text.of(source.words).words
        return source, Corpus.words(keyed, self.data.grouping, self.data.rate)

    def domains(self, held: dict[str, list[Run]]) -> tuple[dict[str, Split], pd.DataFrame]:
        """Subject 0's validation domains, and the ruler domain's words in its split's order."""
        source, words = self.load([r for group in held.values() for r in group])
        ruler = words[words["run"].isin([r.name for r in held[RULER]])]
        return Corpus.domains(source, words, held, self.data), ruler

    def context(self, split: Split) -> Float[Tensor, "n v"]:
        return self.evaluator.logits(self.model, split)[Head.CONTEXT]

    def echo(
        self, reference: pd.DataFrame, pooled: Run, trained: Run
    ) -> tuple[Float[Tensor, "n v"], Bool[Tensor, " n"], float]:
        """One other subject's context logits at each of `reference`'s words (0 where it did not hear the word),
        which words it heard, and its reliability on its `trained` run."""
        source, words = self.load([pooled, trained])
        runs = {run: words[words["run"] == run.name] for run in (pooled, trained)}
        splits = {run: Corpus.split(source, rows, self.data, np.ones(len(rows))) for run, rows in runs.items()}
        at = torch.from_numpy(Text(reference).matches(Text(runs[pooled]))).to(self.evaluator.device)
        logits = self.context(splits[pooled])
        aligned = torch.where((at >= 0)[:, None], logits[at.clamp(min=0)], 0.0)
        labelled = splits[trained].labelled()
        score = self.evaluator.evaluate(
            self.context(splits[trained])[labelled], splits[trained].y[labelled], self.log_prior
        )
        return aligned, at >= 0, score.bacc10

    def echoes(self, reference: pd.DataFrame, runs: list[Run]) -> Echoes:
        """Every pooled subject's `echo`, in subject order."""
        pooled = sorted((r for r in runs if r.partition is Partition.POOL), key=lambda r: r.subject)
        trained = {r.subject: r for r in runs if r.key == TRAINED}
        echoes = [self.echo(reference, run, trained[run.subject]) for run in pooled]
        reliability = torch.tensor([bacc for *_, bacc in echoes], device=self.evaluator.device)
        LOG.info(
            "reliability %s", {run.subject: round(float(r), 4) for run, r in zip(pooled, reliability, strict=True)}
        )
        return Echoes(torch.stack([e[0] for e in echoes]), torch.stack([e[1] for e in echoes]), reliability)

    def macro(self, domains: dict[str, Split], context: dict[str, Float[Tensor, "n v"]]) -> Score:
        """The selection macro over domains' unseen spans, from each domain's context logits."""
        novel = {name: split.novel(self.evaluator.seen) for name, split in domains.items()}
        counted = {name: words for name, words in novel.items() if len(words) >= self.evaluator.min_novel}
        return Score.mean(
            self.evaluator.evaluate(context[name][words], domains[name].y[words], self.log_prior)
            for name, words in counted.items()
        )

    def matched(self, split: Split, context: Float[Tensor, "n v"], present: Bool[Tensor, " n"]) -> Score:
        """The ruler domain's unseen-span words that some other subject heard, scored alone."""
        novel = split.novel(self.evaluator.seen)
        words = novel[present[novel]]
        return self.evaluator.evaluate(context[words], split.y[words], self.log_prior)


def main() -> None:
    """Log the ruler domain and the macro with subject 0 alone and pooled, for every weighting and `w0`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", help="a trained model under models/")
    parser.add_argument("--grouping", type=Grouping, choices=list(Grouping), default=DataConfig().grouping)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    root = DataRoot.from_config()
    pooling = Pooling.of(root, args.name, args.grouping, torch.device("cuda"))
    runs = Run.discover(root.raw())
    domains, reference = pooling.domains(pooling.data.held(runs))
    context = {name: pooling.context(split) for name, split in domains.items()}
    echoes = pooling.echoes(reference, runs)
    heard = echoes.present.any(dim=0)
    LOG.info("ruler words %d, heard by another subject %d", len(heard), int(heard.sum()))
    for label, weighting in WEIGHTINGS.items():
        for w0 in WEIGHTS:
            pool = Pool(w0, weighting)
            pooled = pool.combine(context[RULER], echoes.logits, echoes.present, echoes.reliability)
            matched = pooling.matched(domains[RULER], pooled, heard)
            ruler = pooling.matched(domains[RULER], pooled, torch.ones_like(heard))
            macro = pooling.macro(domains, context | {RULER: pooled})
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
