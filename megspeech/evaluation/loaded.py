"""A trained model, or several averaged as one, loaded to score as their training scored them."""

from dataclasses import dataclass, fields

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor, nn

from core.paths import DataRoot
from megspeech.data.batching.split import Split
from megspeech.data.config import DataConfig
from megspeech.data.raw.runs import Partition, Run
from megspeech.data.spans.spans import Grouping
from megspeech.evaluation.metric.balance import Sinkhorn
from megspeech.evaluation.metric.ruler import Graded
from megspeech.evaluation.reader import Reader
from megspeech.evaluation.scoring.evaluator import Evaluator
from megspeech.evaluation.scoring.score import Epoch, Head, Score, SelectionConfig
from megspeech.model.ensemble import Ensemble
from megspeech.model.input.lowpass import LowPass
from megspeech.model.input.subjects import AS_TRAINED, Fallback, Reading, SubjectLayer


@dataclass(frozen=True)
class Scoring:
    """How a loaded model is scored: spans cut by `grouping`, subjects read as `fallback` and presented as `reading`
    says, each scored set reshaped by `balance` and every window read moved by each of `shifts`."""

    grouping: Grouping = DataConfig.grouping
    fallback: Fallback = AS_TRAINED
    reading: Reading = Reading.OWN
    balance: Sinkhorn = Sinkhorn()
    shifts: tuple[int, ...] = (0,)
    aligned: bool = False
    """Re-colour every other subject's recording to subject 0's sensor covariance (`Alignment`) before the model
    reads it: +0.040 on the Broad ruler for a subject-0 model (0.499 -> 0.539, n=25322, megspeech-7gf.56)."""
    adapted: bool = False
    """Re-estimate batch-norm statistics on each scored split (`Evaluator.adapt`); the Kaggle test's would read its own
    MEG, so a submission written this way needs a rules decision (megspeech-7gf.61)."""
    lowpass: float = 0.0
    """Hz above which every window read is zeroed (`LowPass`); 0 reads the full band. ema9998-s0 at 25/18/12/8/4 Hz:
    0.6445/0.6468/0.6389/0.6248/0.5924 (n=2512): the 12-25 Hz top of the 50 Hz cache carries ~0.006 (4te.5)."""
    partition: Partition = Partition.VALIDATION
    """The labelled partition scored (`DataConfig.ruled`): validation selects; test only reports, once, on a model
    already chosen (megspeech-7gf.81)."""
    official: bool = False
    """Score only the official partition's runs (`share` 0): the ruler every model before the 80/10/10 split was
    selected on, so a split model reads on the same scale as they did (megspeech-7gf.81)."""


@dataclass(frozen=True, kw_only=True)
class LoadedModel(Reader):
    """Trained runs averaged as one model, with the prior and evaluator their training scored them by."""

    model: nn.Module
    evaluator: Evaluator
    log_prior: Float[Tensor, " v"]
    names: tuple[str, ...] = ()

    @classmethod
    def of(cls, root: DataRoot, names: list[str], scoring: Scoring, device: torch.device) -> "LoadedModel":
        """The trained runs behind `names` averaged as one model, scored as their training scored them: the prior of
        the draws they trained on, and the selection that counted their domains."""
        directories = [member for name in names for member in Ensemble.trained(root.models(), name)]
        model = Ensemble.load([directory / "model.pt" for directory in directories], scoring.fallback.unseen)
        model = model.to(device)
        for layer in (module for module in model.modules() if isinstance(module, SubjectLayer)):
            layer.shrink(scoring.fallback.keep, scoring.fallback.unseen)
        data = DataConfig.shared(directories, scoring.grouping)
        model = LowPass(model, data.rate, scoring.lowpass) if scoring.lowpass else model
        geometry = data.geometry(root, device, aligned=scoring.aligned)
        reader = Reader(root, data, device, scoring.reading, geometry, scoring.partition, scoring.official)
        sequences, prior = reader.heard()
        evaluator = Evaluator(
            device,
            sequences,
            SelectionConfig.shared(directories).min_novel,
            balance=scoring.balance,
            shifts=scoring.shifts,
            adapted=scoring.adapted,
        )
        return cls(
            **{field.name: getattr(reader, field.name) for field in fields(Reader)},
            model=model,
            evaluator=evaluator,
            log_prior=prior,
            names=tuple(d.name for d in directories),
        )

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
