"""One group of trained runs that share a `READING`, loaded for scoring."""

from dataclasses import fields
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


class Group:
    @staticmethod
    def load(
        root: DataRoot,
        directories: list[Path],
        scoring: Scoring,
        device: torch.device,
        geometry: Geometry | None,
    ) -> ExportedModel:
        """The runs in `directories`, which trained at one `READING`, averaged as one model."""
        data = DataConfig.shared(directories, scoring.grouping)
        model = scoring.network(directories, data.rate, device)
        if geometry is None or scoring.aligned:
            geometry = data.geometry(root, device, aligned=scoring.aligned)
        reader = Reader(root, data, device, scoring.reading, geometry, scoring.partition)
        sequences, prior = Heard.under(root, data, device)
        evaluator = Evaluator(
            device,
            sequences,
            SelectionConfig.shared(directories).min_novel,
            balance=scoring.balance,
            shifts=scoring.shifts,
            adapted=scoring.adapted,
        )
        return ExportedModel(
            **{field.name: getattr(reader, field.name) for field in fields(Reader)},
            model=model,
            evaluator=evaluator,
            log_prior=prior,
            names=tuple(d.name for d in directories),
        )
