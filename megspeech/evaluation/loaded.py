"""A trained model, or several averaged as one, loaded to score as their training scored them."""

from dataclasses import dataclass

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor, nn

from megspeech.data.batching.split import Split
from megspeech.data.raw.runs import Run
from megspeech.evaluation.metric.ruler import Graded
from megspeech.evaluation.reader import Reader
from megspeech.evaluation.scoring.evaluator import Evaluator
from megspeech.evaluation.scoring.score import Epoch, Head, Score


@dataclass(frozen=True, kw_only=True)
class LoadedModel(Reader):
    """Trained runs averaged as one model, with the prior and evaluator their training scored them by."""

    model: nn.Module
    evaluator: Evaluator
    log_prior: Float[Tensor, " v"]
    names: tuple[str, ...] = ()

    def context(self, split: Split) -> Float[Tensor, "n v"]:
        return self.evaluator.logits(self.model, split)[Head.CONTEXT]

    def labelled(self, split: Split) -> Score:
        """The context head on every labelled word of `split`."""
        labelled = split.labelled()
        return self.evaluator.evaluate(self.context(split)[labelled], split.y[labelled], self.log_prior)

    def unseen(self, run: Run) -> Score:
        """The context head on `run`'s unseen-span words, read from its own MEG."""
        _, split = self.splits([run])[run]
        return self.evaluator.unseen(self.context(split), split, self.log_prior)

    def report(self, runs: list[Run]) -> Epoch:
        """Every domain of `runs` in the scored partition, scored alone, and the macro that selects."""
        domains, _ = self.domains(self.ruled(runs))
        return self.evaluator.report(self.model, domains, self.log_prior, 0, 0.0)

    def graded(self, runs: list[Run], tau: float) -> Graded:
        """Every counted domain of `runs` in the scored partition: its unseen-span words' hits at `tau`."""
        domains, _ = self.domains(self.ruled(runs))
        return self.evaluator.graded(self.model, domains, self.log_prior, tau)

    def probabilities(self, split: Split, tau: float) -> Float[np.ndarray, "n v"]:
        """Every word of `split`'s probabilities, logit-adjusted by `tau`: the context head's, but the word head's for
        a word alone in its span. The context head on a lone word scores near chance (bAcc@10 0.24 vs the word head's
        0.39 on Sherlock1, 0.23 vs 0.29 on TheMoth, n=1296/1216), and 92 of the Deep Kaggle test's 960 rows are lone."""
        logits = self.evaluator.logits(self.model, split)
        chosen = torch.where(split.lone()[:, None], logits[Head.WORD], logits[Head.CONTEXT])
        adjusted = self.evaluator.adjusted(chosen, self.log_prior, tau)
        return torch.softmax(adjusted, dim=-1).cpu().numpy()
