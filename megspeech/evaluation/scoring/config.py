"""How a loaded model is scored."""

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from megspeech.data.config import DataConfig
from megspeech.data.raw.runs import Partition
from megspeech.data.spans.grouping import Grouping
from megspeech.evaluation.metric.balance import PoolSmoothing, Sinkhorn
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
    smooth: PoolSmoothing = PoolSmoothing()
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
    word_share: float = 0.0
    """Weight of the word head against the context head in the log-probabilities scored and submitted
    (`Evaluator.head`): 0 the context head, 1 the word head, between them their log-probability mix."""
    partition: Partition = Partition.VALIDATION
    """The labelled partition scored (`DataConfig.ruled`): val selects; the local test only reports, once, on a model
    already chosen (megspeech-7gf.81)."""

    def __post_init__(self) -> None:
        if not 0.0 <= self.word_share <= 1.0:
            raise ValueError(f"word_share is a mixing weight in [0, 1], got {self.word_share}")

    @staticmethod
    def members(models: Path, names: list[str]) -> list[Path]:
        """The trained runs behind `names` under `models`, each ensemble expanded to its members."""
        return [member for name in names for member in Ensemble.trained(models, name)]

    def network(self, directories: list[Path], rate: float, device: torch.device) -> nn.Module:
        """The trained runs in `directories` averaged as one model on `device`, its subject layers shrunk to this
        scoring's `fallback` and its input band-limited when `lowpass` asks, `rate` Hz being the cache's."""
        model = Ensemble.load([directory / "model.pt" for directory in directories], self.fallback.unseen).to(device)
        for layer in (module for module in model.modules() if isinstance(module, SubjectLayer)):
            layer.shrink(self.fallback.keep, self.fallback.unseen)
        return LowPass(model, rate, self.lowpass) if self.lowpass else model
