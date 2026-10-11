"""What every training epoch records beyond the validation report, so steering never needs a re-run.

`fit` scores a fixed subset of training spans, about the validation words in number, through the report's own path: the
fit-versus-validation gap per epoch. `neural` scores validation with every word's MEG window swapped for another's of
its domain, onsets untouched (`Split.permuted`): what timing alone earns, so the real score less it is the part MEG
earns. Both read the model the report just read, at the tau its macro selected.
"""

import logging
from dataclasses import dataclass, replace

import torch
from jaxtyping import Float
from torch import Tensor, nn

from megspeech.data.batching.split import Split
from megspeech.data.labels.vocabulary import IGNORE
from megspeech.data.spans.spans import PAD
from megspeech.evaluation.scoring.evaluator import Evaluator
from megspeech.evaluation.scoring.score import Epoch, Head

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class TelemetryConfig:
    """Which per-epoch measurements a run records; each costs about one validation read per epoch."""

    fit: bool = True
    """Score a seeded subset of training spans of about the validation's word count (`Epoch.fit`)."""
    neural: bool = True
    """Score validation with MEG windows shuffled across words (`Epoch.shuffled`)."""
    probs: bool = True
    """Save the kept model's per-word probabilities beside it (`probs.pt`)."""
    seed: int = 0
    """Seed of the training subset and of the window permutation; neither touches the training RNG."""


@dataclass(frozen=True)
class Telemetry:
    """The fixed sets `config` scores every epoch: the training subset and the shuffled validation domains."""

    evaluator: Evaluator
    fitted: Split | None = None
    shuffled: dict[str, Split] | None = None

    @classmethod
    def of(
        cls, evaluator: Evaluator, config: TelemetryConfig, train: Split, validation: dict[str, Split]
    ) -> "Telemetry":
        fitted = cls.subset(train, sum(len(split.labelled()) for split in validation.values()), config.seed)
        shuffled = {name: split.permuted(config.seed) for name, split in validation.items()}
        return cls(evaluator, fitted if config.fit else None, shuffled if config.neural else None)

    @staticmethod
    def subset(train: Split, words: int, seed: int) -> Split:
        """Randomly drawn spans of `train` until they hold `words` labelled words."""
        order = torch.randperm(len(train.index), generator=torch.Generator().manual_seed(seed)).to(train.y.device)
        reads = train.index[order]
        labelled = (reads != PAD) & (train.y[reads.clamp(min=0)] != IGNORE)
        counts = labelled.sum(dim=1).cumsum(0)
        kept = int(torch.searchsorted(counts, torch.tensor(words, device=counts.device)).item()) + 1
        return train.spanned(order[:kept])

    def measured(self, model: nn.Module, row: Epoch, log_prior: Float[Tensor, " v"]) -> Epoch:
        """`row` with the measurements this telemetry holds sets for, scored at `row`'s tau."""
        fit = self.fit(model, row, log_prior) if self.fitted is not None else None
        shuffled = self.neural(model, row, log_prior) if self.shuffled is not None else None
        measured = replace(row, fit=fit, shuffled=shuffled)
        if fit is not None or shuffled is not None:
            LOG.info("epoch %d %s", row.epoch, measured.steering())
        return measured

    def fit(self, model: nn.Module, row: Epoch, log_prior: Float[Tensor, " v"]) -> float | None:
        """The context head's balanced top-k on the training subset's labelled words at `row`'s tau."""
        split = self.fitted
        if split is None:
            return None
        words = split.heard()
        context = self.evaluator.logits(model, split)[Head.CONTEXT]
        return self.evaluator.evaluate(context[words], split.y[words], log_prior).by_tau[row.tau]

    def neural(self, model: nn.Module, row: Epoch, log_prior: Float[Tensor, " v"]) -> float | None:
        """The selection macro at `row`'s tau with every domain's MEG windows shuffled across its words."""
        if self.shuffled is None:
            return None
        return self.evaluator.point(model, self.shuffled, self.evaluator.calibration(log_prior, row.tau))
