"""A trained model, or several averaged as one, loaded to score as their training scored them."""

from dataclasses import dataclass

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor, nn

from megspeech.data.batching.split import KAGGLE_FEATURES, Split
from megspeech.data.cache.kaggle import Track
from megspeech.data.corpus import Corpus
from megspeech.data.raw.runs import Run
from megspeech.evaluation.metric.ruler import Graded
from megspeech.evaluation.reader import Reader
from megspeech.evaluation.scoring.evaluator import Evaluator
from megspeech.evaluation.scoring.probabilities import Probabilities
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

    @property
    def adapted_on(self) -> int:
        """The Sherlock1 session the model trained on as `DataConfig.adapt`; 0 for none."""
        return self.data.adapt

    def run_context(self, run: Run) -> tuple[Split, Float[Tensor, "n v"]]:
        """`run`'s split, read from its own MEG, and the context head's log-probabilities for its words."""
        _, split = self.splits([run])[run]
        return split, self.context(split)

    def unseen(self, run: Run) -> Score:
        """The context head on `run`'s unseen-span words, read from its own MEG."""
        split, context = self.run_context(run)
        return self.evaluator.unseen(context, split, self.log_prior)

    def domain_logits(self, runs: list[Run]) -> tuple[dict[str, Split], dict[str, dict[Head, Float[Tensor, "n v"]]]]:
        """Every domain of `runs` in the scored partition as the model reads it, and both heads' log-probabilities."""
        domains, _ = self.domains(self.ruled(runs))
        return domains, {name: self.evaluator.logits(self.model, split) for name, split in domains.items()}

    def report(self, runs: list[Run]) -> Epoch:
        """Every domain of `runs` in the scored partition, scored alone, and the macro that selects."""
        domains, logits = self.domain_logits(runs)
        return self.evaluator.summarised(logits, domains, self.log_prior, 0, 0.0)

    def graded(self, runs: list[Run], tau: float) -> Graded:
        """Every counted domain of `runs` in the scored partition: its unseen-span words' hits at `tau`."""
        domains, logits = self.domain_logits(runs)
        return self.evaluator.graded_logits(logits, domains, self.log_prior, tau)

    def probabilities_of(self, runs: list[Run], tau: float) -> Probabilities:
        """Every labelled word of `runs`' scored domains with its probabilities at `tau`."""
        domains, logits = self.domain_logits(runs)
        return Probabilities.from_logits(self.evaluator, logits, domains, self.log_prior, tau)

    def kaggle(self, track: Track) -> Split:
        """Every row of `track`'s Kaggle test as this model reads it, each from its own subject's MEG."""
        kaggle = track.kaggle(self.root.raw())
        windows = Corpus.kaggle(self.root, self.data, track, self.device, self.geometry)
        return self.read(f"{KAGGLE_FEATURES}-{track}", Split.from_kaggle(kaggle, windows))

    def chosen(self, split: Split) -> Float[Tensor, "n v"]:
        """Every word of `split`'s log-probabilities: the context head's, but the word head's for a word alone in its
        span. The context head on a lone word scores near chance (bAcc@10 0.24 vs the word head's 0.39 on Sherlock1,
        0.23 vs 0.29 on TheMoth, n=1296/1216), and 92 of the Deep Kaggle test's 960 rows are lone."""
        logits = self.evaluator.logits(self.model, split)
        return torch.where(split.lone()[:, None], logits[Head.WORD], logits[Head.CONTEXT])

    def kaggle_chosen(self, track: Track) -> tuple[Split, Float[Tensor, "n v"]]:
        """`track`'s Kaggle split and its words' `chosen` log-probabilities."""
        split = self.kaggle(track)
        return split, self.chosen(split)

    def finished(self, chosen: Float[Tensor, "n v"], tau: float) -> Float[np.ndarray, "n v"]:
        """`chosen` log-probabilities logit-adjusted by `tau` and normalised."""
        adjusted = self.evaluator.adjusted(chosen, self.log_prior, tau)
        return torch.softmax(adjusted, dim=-1).cpu().numpy()

    def probabilities(self, split: Split, tau: float) -> Float[np.ndarray, "n v"]:
        """Every word of `split`'s probabilities, logit-adjusted by `tau`."""
        return self.finished(self.chosen(split), tau)

    def kaggle_probabilities(self, track: Track, tau: float) -> Float[np.ndarray, "n v"]:
        """Every row of `track`'s Kaggle test, as `probabilities` reads a split."""
        _, chosen = self.kaggle_chosen(track)
        return self.finished(chosen, tau)
