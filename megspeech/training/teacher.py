"""Who the context head learns from besides the labels: nobody, or a trained ensemble's distribution on every
training word (knowledge distillation, Hinton et al. 2015; born-again networks, Furlanello et al. 2018)."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import override

import torch

from core.paths import DataRoot
from megspeech.data.batching.split import Split
from megspeech.evaluation.scoring.evaluator import Evaluator
from megspeech.evaluation.scoring.score import Head
from megspeech.model.ensemble import Ensemble
from megspeech.objective.term import ContextTerm, DistilledTerm
from megspeech.training.config import OptimConfig


class Teacher(ABC):
    """What turns the context head's label term into the term a run trains."""

    @abstractmethod
    def taught(self, context: ContextTerm, train: Split, evaluator: Evaluator) -> ContextTerm: ...

    @staticmethod
    def of(optim: OptimConfig, root: DataRoot) -> "Teacher":
        """Nobody without a `teacher` run; that run under `root`'s models otherwise."""
        return Ensembled(root.models(), optim.teacher, optim.distil) if optim.teacher else Untaught()


class Untaught(Teacher):
    """The labels alone."""

    @override
    def taught(self, context: ContextTerm, train: Split, evaluator: Evaluator) -> ContextTerm:
        return context


@dataclass(frozen=True)
class Ensembled(Teacher):
    """The trained run `name` under `models` (an ensemble's members averaged), its context head read once by
    `evaluator` over the training words."""

    models: Path
    name: str
    share: float

    @override
    def taught(self, context: ContextTerm, train: Split, evaluator: Evaluator) -> ContextTerm:
        paths = [directory / "model.pt" for directory in Ensemble.trained(self.models, self.name)]
        model = Ensemble.load(paths).to(evaluator.device)
        with torch.no_grad():
            taught = evaluator.logits(model, train)[Head.CONTEXT]
        return DistilledTerm(**vars(context), taught=taught, y=train.y, share=self.share)
