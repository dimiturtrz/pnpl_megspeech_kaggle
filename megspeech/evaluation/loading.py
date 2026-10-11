"""A trained ensemble loaded for scoring: the model, prior and evaluator that result."""

import torch

from core.paths import DataRoot
from megspeech.data.cache.alignment import Geometry
from megspeech.data.config import DataConfig
from megspeech.evaluation.blended import Blended
from megspeech.evaluation.exported import ExportedModel
from megspeech.evaluation.group import Group
from megspeech.evaluation.scoring.config import Scoring
from megspeech.evaluation.scoring.score import SelectionConfig


class Loader:
    @staticmethod
    def of(
        root: DataRoot,
        names: list[str],
        scoring: Scoring,
        device: torch.device,
        geometry: Geometry | None = None,
    ) -> ExportedModel:
        """The trained runs behind `names` averaged as one model, scored as their training scored them: the prior of
        the draws they trained on, and the selection that counted their domains. A `geometry` already built in this
        process stands in for the one `scoring` would build, unless it asks for `aligned`."""
        directories = Scoring.members(root.models(), names)
        SelectionConfig.shared(directories)
        group = Group(root, scoring, device, geometry)
        groups = [group.load(members) for members in DataConfig.partitioned(directories)]
        return groups[0] if len(groups) == 1 else Blended.of(groups)
