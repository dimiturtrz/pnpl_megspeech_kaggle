"""A loaded model's context head fused with a word-level log-probability source on a grid of weights, per domain."""

import logging
from dataclasses import dataclass
from typing import Protocol

from jaxtyping import Float
from torch import Tensor

from megspeech.data.batching.split import Split
from megspeech.data.raw.runs import Run
from megspeech.evaluation.loaded.model import LoadedModel
from megspeech.evaluation.scoring.score import Head

LOG = logging.getLogger(__name__)


class Source(Protocol):
    """Every word of a split's log-probability over the vocabulary, given the context head's logits."""

    def logp(self, split: Split, context: Float[Tensor, "n v"]) -> Float[Tensor, "n v"]: ...


@dataclass(frozen=True)
class Fusion:
    """A loaded model's domains read with and without `source`, named `label`: `context + weight * source` at every
    one of `weights`."""

    loaded: LoadedModel
    source: Source
    weights: tuple[float, ...]
    label: str

    def domain(self, split: Split) -> dict[float, float]:
        """BAcc@10 on `split`'s unseen-span words at every fusion weight."""
        evaluator = self.loaded.evaluator
        context = evaluator.logits(self.loaded.model, split)[Head.CONTEXT]
        source = self.source.logp(split, context)
        novel = split.novel(evaluator.seen)
        alone = evaluator.evaluate(source[novel], split.y[novel], self.loaded.log_prior)
        LOG.info("%s alone %.4f over %d words", self.label, alone.bacc10, alone.n)
        return {
            weight: evaluator.evaluate((context + weight * source)[novel], split.y[novel], self.loaded.log_prior).bacc10
            for weight in self.weights
        }

    def table(self, held: dict[str, list[Run]]) -> dict[str, dict[float, float]]:
        """Every counted domain's scores by fusion weight."""
        domains, _ = self.loaded.domains(held)
        evaluator = self.loaded.evaluator
        return {name: self.domain(split) for name, split in evaluator.eligible(domains).items()}
