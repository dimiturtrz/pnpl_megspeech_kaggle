"""A loaded model read for export: its Kaggle test rows and the per-word probabilities it is submitted or saved by."""

from dataclasses import dataclass

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor

from megspeech.data.batching.split import KAGGLE_FEATURES, Split
from megspeech.data.cache.kaggle import Track
from megspeech.data.corpus import Corpus
from megspeech.data.raw.runs import Run
from megspeech.evaluation.loaded import LoadedModel
from megspeech.evaluation.scoring.probabilities import Probabilities
from megspeech.evaluation.scoring.score import Head


@dataclass(frozen=True, kw_only=True)
class ExportedModel(LoadedModel):
    """A `LoadedModel` that also reads the Kaggle test and writes probabilities."""

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
