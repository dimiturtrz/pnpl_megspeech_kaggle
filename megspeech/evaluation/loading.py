"""A trained ensemble loaded for scoring: the model, prior and evaluator that result."""

from dataclasses import fields

import torch

from core.paths import DataRoot
from megspeech.data.cache.alignment import Geometry
from megspeech.data.config import DataConfig
from megspeech.data.heard import Heard
from megspeech.evaluation.loaded import LoadedModel
from megspeech.evaluation.reader import Reader
from megspeech.evaluation.scoring.config import Scoring
from megspeech.evaluation.scoring.evaluator import Evaluator
from megspeech.evaluation.scoring.score import SelectionConfig


class Loader:
    @staticmethod
    def of(
        root: DataRoot,
        names: list[str],
        scoring: Scoring,
        device: torch.device,
        geometry: Geometry | None = None,
    ) -> LoadedModel:
        """The trained runs behind `names` averaged as one model, scored as their training scored them: the prior of
        the draws they trained on, and the selection that counted their domains. A `geometry` already built in this
        process stands in for the one `scoring` would build, unless it asks for `aligned`."""
        directories = Scoring.members(root.models(), names)
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
        return LoadedModel(
            **{field.name: getattr(reader, field.name) for field in fields(Reader)},
            model=model,
            evaluator=evaluator,
            log_prior=prior,
            names=tuple(d.name for d in directories),
        )
