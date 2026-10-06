"""A trained ensemble loaded for scoring: how it is scored, and the model, prior and evaluator that result."""

from dataclasses import dataclass, fields

import torch

from core.paths import DataRoot
from megspeech.data.config import DataConfig
from megspeech.data.raw.runs import Partition
from megspeech.data.spans.spans import Grouping
from megspeech.evaluation.loaded import LoadedModel
from megspeech.evaluation.metric.balance import Sinkhorn
from megspeech.evaluation.reader import Reader
from megspeech.evaluation.scoring.evaluator import Evaluator
from megspeech.evaluation.scoring.score import SelectionConfig
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
    """The labelled partition scored (`DataConfig.ruled`): val selects; the local test only reports, once, on a model
    already chosen (megspeech-7gf.81)."""


class Loader:
    @staticmethod
    def of(root: DataRoot, names: list[str], scoring: Scoring, device: torch.device) -> LoadedModel:
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
        reader = Reader(root, data, device, scoring.reading, geometry, scoring.partition)
        sequences, prior = reader.heard()
        evaluator = Evaluator(
            device,
            sequences,
            SelectionConfig.shared(directories).min_novel,
            balance=scoring.balance,
            shifts=scoring.shifts,
            adapted=scoring.adapted,
        )
        return LoadedModel(
            **{field.name: getattr(reader, field.name) for field in fields(Reader)},
            model=model,
            evaluator=evaluator,
            log_prior=prior,
            names=tuple(d.name for d in directories),
        )
