"""Score a model on validation: both heads per domain, raw and logit-adjusted, and the macro that selects.

Balanced accuracy rewards rare words as much as frequent ones, so scores are read both raw and logit-adjusted
(Menon et al. 2021): subtracting the log training prior turns p(word | MEG) into a score proportional to
p(MEG | word), the Bayes rule for a balanced metric.
"""

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import torch
from jaxtyping import Float, Int
from torch import Tensor, nn

from megspeech.data.spans import PAD
from megspeech.data.split import EVAL_SPANS, Split
from megspeech.data.vocabulary import WORDS
from megspeech.evaluation.ruler import Coverage, Graded, Interval, Ruler

ADJUSTMENTS = (0.0, 0.5, 1.0)
"""Logit-adjustment strengths tried on validation: none, half, and the full Bayes correction."""

MIN_NOVEL = 100
"""Unseen-span words a domain needs to count toward selection: two per class of the 50 on average."""


class Head(StrEnum):
    CONTEXT = "context"
    WORD = "word"


@dataclass(frozen=True)
class Score:
    tau: float
    bacc10: float
    by_tau: dict[float, float]
    n: int

    @classmethod
    def best(cls, by_tau: dict[float, float], n: int) -> "Score":
        tau = max(by_tau, key=by_tau.__getitem__)
        return cls(tau, by_tau[tau], by_tau, n)


@dataclass(frozen=True)
class Report:
    """One validation domain: both heads, the context head on holdout-like and on unseen spans, and its top-1."""

    context: Score
    word: Score
    dense: Score
    novel: Score
    bacc1: float
    coverage: Coverage


@dataclass(frozen=True)
class Epoch:
    """One validation pass over every domain, the macro over domains' unseen spans that selects, and its interval."""

    epoch: int
    loss: float
    domains: dict[str, Report]
    macro: Score
    interval: Interval


@dataclass(frozen=True)
class Evaluator:
    """Scores a model on validation domains; `seen` marks the word orders training heard, `min_novel` who counts."""

    device: torch.device
    seen: frozenset[tuple[int, ...]] = frozenset()
    min_novel: int = MIN_NOVEL
    ruler: Ruler = Ruler()

    def logits(self, model: nn.Module, split: Split) -> dict[Head, Float[Tensor, "n v"]]:
        """Both heads' logits for every word of `split`, in the split's word order.

        Every batch reserves room for full spans, so a compiled model sees one shape (and the last batch's).
        """
        model.eval()
        words, rows = len(split.x), EVAL_SPANS * split.index.shape[1]
        out = {head: torch.zeros(words + 1, WORDS, device=self.device) for head in Head}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for spans in torch.arange(len(split.index), device=self.device).split(EVAL_SPANS):
                batch = split.batch(spans, rows)
                context, word = model(batch.x, batch.onset, batch.slot)
                at = torch.where(batch.index != PAD, batch.index, words)
                out[Head.CONTEXT][at] = context.float()
                out[Head.WORD][at] = word.float()
        return {head: logits[:words] for head, logits in out.items()}

    def evaluate(
        self, logits: Float[Tensor, "n v"], labels: Int[Tensor, " n"], log_prior: Float[Tensor, " v"]
    ) -> Score:
        by_tau = {
            tau: self.ruler.balanced(self.ruler.hits(logits - tau * log_prior, labels), labels) for tau in ADJUSTMENTS
        }
        return Score.best(by_tau, len(labels))

    def assess(self, model: nn.Module, split: Split, log_prior: Float[Tensor, " v"]) -> Report:
        logits = self.logits(model, split)
        dense = split.dense()
        novel = split.novel(self.seen)
        top1 = Ruler(k=1)
        return Report(
            self.evaluate(logits[Head.CONTEXT], split.y, log_prior),
            self.evaluate(logits[Head.WORD], split.y, log_prior),
            self.evaluate(logits[Head.CONTEXT][dense], split.y[dense], log_prior),
            self.evaluate(logits[Head.CONTEXT][novel], split.y[novel], log_prior),
            top1.balanced(top1.hits(logits[Head.CONTEXT], split.y), split.y),
            Coverage.of(split.y[novel]),
        )

    def graded(self, model: nn.Module, domains: dict[str, Split], log_prior: Float[Tensor, " v"], tau: float) -> Graded:
        """The context head's per-word hits on the unseen spans of every domain that counts toward selection."""
        novel = {name: split.novel(self.seen) for name, split in domains.items()}
        counted = {name: split for name, split in domains.items() if len(novel[name]) >= self.min_novel}
        return Graded(
            {
                name: (
                    self.ruler.hits(
                        self.logits(model, split)[Head.CONTEXT][novel[name]] - tau * log_prior, split.y[novel[name]]
                    ),
                    split.y[novel[name]],
                )
                for name, split in counted.items()
            }
        )

    def report(
        self, model: nn.Module, domains: dict[str, Split], log_prior: Float[Tensor, " v"], epoch: int, loss: float
    ) -> Epoch:
        """Every domain scored alone; selection reads the macro over unseen spans of domains with enough of them."""
        reports = {name: self.assess(model, split, log_prior) for name, split in domains.items()}
        counted = [r.novel for r in reports.values() if r.novel.n >= self.min_novel]
        by_tau = {tau: float(np.mean([score.by_tau[tau] for score in counted])) for tau in ADJUSTMENTS}
        macro = Score.best(by_tau, sum(score.n for score in counted))
        interval = self.ruler.interval(self.ruler.macro(self.graded(model, domains, log_prior, macro.tau)))
        return Epoch(epoch, loss, reports, macro, interval)
