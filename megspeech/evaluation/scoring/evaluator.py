"""Score a model on validation: both heads per domain, raw and logit-adjusted, and the macro that selects.

Balanced accuracy rewards rare words as much as frequent ones, so scores are read both raw and logit-adjusted
(Menon et al. 2021): subtracting the log training prior turns p(word | MEG) into a score proportional to
p(MEG | word), the Bayes rule for a balanced metric.
"""

import copy
from dataclasses import dataclass, replace

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn

from megspeech.data.batching.split import EVAL_SPANS, Split
from megspeech.data.labels.vocabulary import VOCAB_SIZE
from megspeech.data.spans.spans import PAD
from megspeech.evaluation.metric.balance import Sinkhorn
from megspeech.evaluation.metric.ruler import Coverage, Graded, Ruler
from megspeech.evaluation.scoring.score import ADJUSTMENTS, MIN_NOVEL, Epoch, Head, Report, Score
from megspeech.model.words.layers.norm import MaskedBatchNorm1d


@dataclass(frozen=True)
class Evaluator:
    """Scores a model on validation domains; `seen` marks the word orders training heard, `min_novel` who counts,
    `balance` reshapes each scored set's adjusted scores before their top-k is read."""

    device: torch.device
    seen: frozenset[tuple[int, ...]] = frozenset()
    min_novel: int = MIN_NOVEL
    ruler: Ruler = Ruler()
    balance: Sinkhorn = Sinkhorn()
    shifts: tuple[int, ...] = (0,)
    """Samples every window is read moved by; the reads' log-probabilities are averaged (test-time augmentation)."""
    adapted: bool = False
    """Read every split through a copy of the model whose batch-norm statistics are the split's own (`adapt`)."""

    def logits(self, model: nn.Module, split: Split) -> dict[Head, Float[Tensor, "n v"]]:
        """Both heads' log-probabilities for every word of `split`, averaged over the `shifts`, in word order."""
        model = self.adapt(model, split) if self.adapted else model
        reads = [self.read(model, replace(split, windows=split.windows.shifted(shift))) for shift in self.shifts]
        return {head: torch.stack([read[head].log_softmax(-1) for read in reads]).mean(0) for head in Head}

    def adapt(self, model: nn.Module, split: Split) -> nn.Module:
        """A copy of `model` whose batch-norm statistics are re-estimated on `split`'s unlabelled windows, every
        other layer left in evaluation: AdaBN (Li et al. 2016) against the session shift."""
        adapted = copy.deepcopy(model).eval()
        for norm in (module for module in adapted.modules() if isinstance(module, MaskedBatchNorm1d)):
            norm.restart()
            norm.train()
        self.sweep(adapted, split)
        return adapted.eval()

    def read(self, model: nn.Module, split: Split) -> dict[Head, Float[Tensor, "n v"]]:
        """Both heads' logits for every word of `split`, in the split's word order."""
        model.eval()
        return self.sweep(model, split)

    def sweep(self, model: nn.Module, split: Split) -> dict[Head, Float[Tensor, "n v"]]:
        """`read` in whatever mode `model` is in.

        Every batch reserves room for full spans, so a compiled model sees one shape (and the last batch's).
        """
        words, rows = len(split.y), EVAL_SPANS * split.index.shape[1]
        out = {head: torch.zeros(words + 1, VOCAB_SIZE, device=self.device) for head in Head}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for spans in torch.arange(len(split.index), device=self.device).split(EVAL_SPANS):
                batch = split.batch(spans, rows)
                context, word = model(batch.x, batch.onset, batch.slot, batch.subject)
                at = torch.where(batch.index != PAD, batch.index, words)
                out[Head.CONTEXT][at] = context.float()
                out[Head.WORD][at] = word.float()
        return {head: logits[:words] for head, logits in out.items()}

    def evaluate(
        self, logits: Float[Tensor, "n v"], labels: Int[Tensor, " n"], log_prior: Float[Tensor, " v"]
    ) -> Score:
        by_tau = {
            tau: self.ruler.balanced(self.ruler.hits(self.adjusted(logits, log_prior, tau), labels), labels)
            for tau in ADJUSTMENTS
        }
        return Score.best(by_tau, len(labels))

    def adjusted(
        self, logits: Float[Tensor, "n v"], log_prior: Float[Tensor, " v"], tau: float
    ) -> Float[Tensor, "n v"]:
        """`logits` logit-adjusted by `tau`, then balanced over the set."""
        return self.balance(logits - tau * log_prior)

    def unseen(self, context: Float[Tensor, "n v"], split: Split, log_prior: Float[Tensor, " v"]) -> Score:
        """The context head on `split`'s unseen-span words."""
        novel = split.novel(self.seen)
        return self.evaluate(context[novel], split.y[novel], log_prior)

    def counted[K](self, scores: dict[K, Score]) -> dict[K, Score]:
        """The `scores` over enough unseen-span words to count toward selection."""
        return {key: score for key, score in scores.items() if score.n >= self.min_novel}

    def scored(self, logits: dict[Head, Float[Tensor, "n v"]], split: Split, log_prior: Float[Tensor, " v"]) -> Report:
        """One domain's report from its already computed logits: every labelled word scored, out-of-vocabulary
        words only lending their spans context."""
        labelled, dense, novel = split.labelled(), split.kaggle_like(), split.novel(self.seen)
        context, word, y = logits[Head.CONTEXT][labelled], logits[Head.WORD][labelled], split.y[labelled]
        unseen, novel_y = logits[Head.CONTEXT][novel], split.y[novel]
        adjusted = {tau: self.adjusted(unseen, log_prior, tau) for tau in ADJUSTMENTS}
        top1 = Ruler(k=1)
        return Report(
            self.evaluate(context, y, log_prior),
            self.evaluate(word, y, log_prior),
            self.evaluate(logits[Head.CONTEXT][dense], split.y[dense], log_prior),
            self.evaluate(unseen, novel_y, log_prior),
            {tau: top1.balanced(top1.hits(scores, novel_y), novel_y) for tau, scores in adjusted.items()},
            {
                tau: Ruler.information(scores.argmax(dim=1), novel_y, log_prior.exp())
                for tau, scores in adjusted.items()
            },
            Coverage.of(novel_y),
        )

    def graded(self, model: nn.Module, domains: dict[str, Split], log_prior: Float[Tensor, " v"], tau: float) -> Graded:
        """The context head's per-word hits on the unseen spans of every domain that counts toward selection."""
        counted = {name: split for name, split in domains.items() if len(split.novel(self.seen)) >= self.min_novel}
        context = {name: self.logits(model, split)[Head.CONTEXT] for name, split in counted.items()}
        return self.grade(context, counted, log_prior, tau)

    def grade(
        self,
        context: dict[str, Float[Tensor, "n v"]],
        domains: dict[str, Split],
        log_prior: Float[Tensor, " v"],
        tau: float,
    ) -> Graded:
        """`graded` from each counted domain's already computed context logits."""
        novel = {name: domains[name].novel(self.seen) for name in context}
        return Graded(
            {
                name: (
                    self.ruler.hits(self.adjusted(logits[novel[name]], log_prior, tau), domains[name].y[novel[name]]),
                    domains[name].y[novel[name]],
                )
                for name, logits in context.items()
            }
        )

    def report(
        self, model: nn.Module, domains: dict[str, Split], log_prior: Float[Tensor, " v"], epoch: int, loss: float
    ) -> Epoch:
        """Every domain scored alone; selection reads the macro over unseen spans of domains with enough of them.

        Each domain's logits are computed once and serve both its report and the interval's per-word hits.
        """
        logits = {name: self.logits(model, split) for name, split in domains.items()}
        reports = {name: self.scored(logits[name], split, log_prior) for name, split in domains.items()}
        counted = self.counted({name: r.novel for name, r in reports.items()})
        macro = Score.mean(counted.values())
        context = {name: logits[name][Head.CONTEXT] for name in counted}
        interval = self.ruler.interval(self.ruler.macro(self.grade(context, domains, log_prior, macro.tau)))
        chosen = [reports[name] for name in counted]
        top1 = Score.average(r.top1[macro.tau] for r in chosen)
        bits = Score.average(r.bits[macro.tau] for r in chosen)
        return Epoch(epoch, loss, reports, macro, interval, top1, bits)
