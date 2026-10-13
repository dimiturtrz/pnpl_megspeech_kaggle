"""One group of trained runs that share a `READING`, loaded for scoring."""

from dataclasses import dataclass, fields
from pathlib import Path

import torch

from core.paths import DataRoot
from megspeech.data.cache.alignment import Geometry
from megspeech.data.config import DataConfig
from megspeech.data.heard import Heard
from megspeech.evaluation.exported import ExportedModel
from megspeech.evaluation.reader import Reader
from megspeech.evaluation.scoring.config import Scoring
from megspeech.evaluation.scoring.evaluator import Evaluator
from megspeech.evaluation.scoring.score import SelectionConfig


@dataclass(frozen=True)
class Group:
    """Where groups load from, how they are scored, on which device; a `geometry` already built in this process
    stands in for the one `scoring` would build, unless it asks for `aligned`."""

    root: DataRoot
    scoring: Scoring
    device: torch.device
    geometry: Geometry | None = None

    def load(self, directories: list[Path]) -> ExportedModel:
        """The runs in `directories`, which trained at one `READING`, averaged as one model."""
        scoring = self.scoring
        data = DataConfig.shared(directories, scoring.grouping)
        model = scoring.network(directories, data.rate, self.device)
        geometry = self.geometry
        if geometry is None or scoring.aligned:
            geometry = data.geometry(self.root, self.device, aligned=scoring.aligned)
        reader = Reader(self.root, data, self.device, scoring.reading, geometry, scoring.partition)
        sequences, prior = Heard.under(self.root, data, self.device)
        evaluator = Evaluator(
            self.device,
            sequences,
            SelectionConfig.shared(directories).min_novel,
            balance=scoring.balance,
            smooth=scoring.smooth,
            shifts=scoring.shifts,
            adapted=scoring.adapted,
            word_share=scoring.word_share,
        )
        return ExportedModel(
            **{field.name: getattr(reader, field.name) for field in fields(Reader)},
            model=model,
            evaluator=evaluator,
            log_prior=prior,
            names=tuple(d.name for d in directories),
        )
