"""Who the context head learns from besides the labels: nobody, or a trained ensemble's distribution on every
training word (knowledge distillation, Hinton et al. 2015; born-again networks, Furlanello et al. 2018)."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import ClassVar, override

import pandas as pd
import torch
from jaxtyping import Float, Int
from torch import Tensor

from core.paths import DataRoot
from megspeech.data.batching.split import Split
from megspeech.data.config import DataConfig
from megspeech.data.raw.runs import Column, Run
from megspeech.evaluation.scoring.evaluator import Evaluator
from megspeech.evaluation.scoring.score import Head
from megspeech.model.ensemble import Ensemble
from megspeech.objective.term import ContextTerm, DistilledTerm


@dataclass(frozen=True)
class TeacherConfig:
    """Who the context head is distilled from; the `optim.distillation` group, its leaves also under their flat
    `--optim.<name>` flags."""

    FLAT: ClassVar[bool] = True

    teacher: str = ""
    """A trained run or ensemble under models/ whose context-head distribution on every labelled training word the
    context head is also trained toward (`megspeech.training.teacher`); empty trains on the labels alone."""
    teacher_folds: int = 1
    """Cross-fitted teacher: `teacher` names `<teacher>-f0 ... <teacher>-f{K-1}`, each trained at `data.crossfit` K with
    `data.dropped` its own fold, and a training word is taught by the model that never trained on its session
    (megspeech-7gf.130); 1 reads `teacher` alone."""
    distil: float = 0.5
    """Share of the context loss that is the KL to the `teacher`'s distribution, the rest the label's; read only
    with a teacher."""
    temperature: float = 1.0
    """Temperature both the teacher's and the student's distributions are softened by in the KL; read only with a
    teacher."""
    unlabelled: bool = False
    """Distil on every training word in a span, the ~60% outside the 50 too, not only the labelled; read only with a
    teacher."""


class Teacher(ABC):
    """What turns the context head's label term into the term a run trains."""

    @abstractmethod
    def taught(self, context: ContextTerm, train: Split, evaluator: Evaluator) -> ContextTerm: ...

    @staticmethod
    def of(config: TeacherConfig, root: DataRoot, data: DataConfig, positions: pd.DataFrame) -> "Teacher":
        """Nobody without a `teacher` run; that run under `root`'s models otherwise, or with `teacher_folds` its fold
        runs, each teaching the words of the `data` sessions it held out.
        `positions` holds each training word's `run`."""
        if not config.teacher:
            return Untaught()
        whole = Ensembled(
            root.models(), config.teacher, config.distil, config.temperature, unlabelled=config.unlabelled
        )
        if config.teacher_folds == 1:
            return whole
        held = positions[Column.RUN].map(data.fold_of(Run.everywhere(root), config.teacher_folds))
        return CrossFitted.over(whole, config.teacher_folds, held)


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
    temperature: float = 1.0
    unlabelled: bool = field(default=False, kw_only=True)

    def read(self, train: Split, evaluator: Evaluator) -> Float[Tensor, "n v"]:
        """The run's context head over every word of `train`."""
        paths = [directory / "model.pt" for directory in Ensemble.trained(self.models, self.name)]
        model = Ensemble.load(paths).to(evaluator.device)
        with torch.no_grad():
            return evaluator.logits(model, train)[Head.CONTEXT]

    @override
    def taught(self, context: ContextTerm, train: Split, evaluator: Evaluator) -> ContextTerm:
        taught = self.read(train, evaluator)
        return DistilledTerm(
            **vars(context),
            taught=taught,
            y=train.y,
            share=self.share,
            temperature=self.temperature,
            unlabelled=self.unlabelled,
        )


@dataclass(frozen=True)
class CrossFitted(Teacher):
    """`whole`'s run read as `<name>-f0 ... <name>-f{K-1}`: each training word's context distribution comes from the
    fold run whose training left its session out, so no target is a memorised one (megspeech-7gf.130)."""

    whole: Ensembled
    held: Int[Tensor, " n"]
    """Each training word's fold, in train-split order."""
    folds: int = 1

    @classmethod
    def over(cls, whole: Ensembled, folds: int, held: "pd.Series[int]") -> "CrossFitted":
        """The teacher over `folds` runs for words in `held` folds; every word must be in one of them."""
        if held.isna().any():
            raise ValueError(f"{int(held.isna().sum())} training words are in no crossfit fold")
        return cls(whole, torch.tensor(held.to_numpy(dtype="int64")), folds)

    def members(self) -> list[Ensembled]:
        return [replace(self.whole, name=f"{self.whole.name}-f{fold}") for fold in range(self.folds)]

    @override
    def taught(self, context: ContextTerm, train: Split, evaluator: Evaluator) -> ContextTerm:
        if len(self.held) != len(train.y):
            raise ValueError(f"{len(self.held)} word folds against {len(train.y)} training words")
        held = self.held.to(evaluator.device)
        reads = torch.stack([member.read(train, evaluator) for member in self.members()])
        taught = reads[held, torch.arange(len(held), device=held.device)]
        return DistilledTerm(
            **vars(context),
            taught=taught,
            y=train.y,
            share=self.whole.share,
            temperature=self.whole.temperature,
            unlabelled=self.whole.unlabelled,
        )
