"""A trained ensemble loaded for scoring: the model, prior and evaluator that result."""

from megspeech.data.config import DataConfig
from megspeech.evaluation.loaded.blended import Blended
from megspeech.evaluation.loaded.exported import ExportedModel
from megspeech.evaluation.loaded.group import Group
from megspeech.evaluation.scoring.config import Scoring
from megspeech.evaluation.scoring.score import SelectionConfig


class Loader:
    @staticmethod
    def of(group: Group, names: list[str]) -> ExportedModel:
        """The trained runs behind `names` averaged as one model, each run group loaded as `group` loads it and
        scored as their training scored them: the prior of the draws they trained on, and the selection that counted
        their domains."""
        directories = Scoring.members(group.root.models(), names)
        SelectionConfig.shared(directories)
        groups = [group.load(members) for members in DataConfig.partitioned(directories)]
        return groups[0] if len(groups) == 1 else Blended.of(groups)
