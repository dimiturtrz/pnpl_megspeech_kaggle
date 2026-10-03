from pathlib import Path
from typing import cast

import pytest
import torch
from torch import nn

from core.paths import DataRoot
from megspeech.data.labels.phonemes import Pronunciation
from megspeech.model.auxiliary import WithAuxiliary
from megspeech.model.config import ContextConfig
from megspeech.model.sentence import SentenceNet
from megspeech.model.words.conv import ConvConfig
from megspeech.objective.auxiliary.lexical import LexicalTerm
from megspeech.objective.auxiliary.phonetic import PhoneticTerm
from megspeech.objective.auxiliary.semantic import SemanticTerm
from megspeech.objective.scorer import DIM
from megspeech.objective.targets.t5 import Lexicon
from megspeech.training.auxiliary import Auxiliary, Joint, NoAuxiliary
from megspeech.training.config import OptimConfig

CPU = torch.device("cpu")
GENERATOR = torch.Generator().manual_seed(0)


class Root:
    def __init__(self, path: Path) -> None:
        self.base = path

    def processed(self) -> Path:
        return self.base


def net() -> SentenceNet:
    return SentenceNet(
        nn.Identity(), ContextConfig(width=16, heads=2, layers=1, conv=ConvConfig(hidden=8, dilations=(1, 2)))
    )


def semantic() -> SemanticTerm:
    return SemanticTerm(2.0, Lexicon(torch.tensor([10, 20]), torch.randn(2, DIM, generator=GENERATOR)))


def phonetic() -> PhoneticTerm:
    return PhoneticTerm(0.5, Pronunciation(torch.tensor([10, 20]), torch.tensor([[3, 1], [7, -1]])))


def identity(model: WithAuxiliary) -> WithAuxiliary:
    return model


def test_no_auxiliary_trained() -> None:
    model = net()
    assert NoAuxiliary().trained(model, identity) is model
    assert NoAuxiliary().terms == ()


def test_joint_trained() -> None:
    model = net()
    trained = Joint((semantic(), phonetic())).trained(model, identity)
    assert isinstance(trained, WithAuxiliary)
    assert trained.net is model
    assert len(trained.heads) == 2


def test_of(tmp_path: Path) -> None:
    root = cast("DataRoot", Root(tmp_path))
    assert isinstance(Auxiliary.of(OptimConfig(), root, CPU), NoAuxiliary)
    with pytest.raises(FileNotFoundError):
        Auxiliary.of(OptimConfig(semantic=1.0), root, CPU)
    with pytest.raises(FileNotFoundError):
        Auxiliary.of(OptimConfig(phonetic=1.0), root, CPU)
    semantic().lexicon.save(Lexicon.path(root))
    phonetic().pronunciation.save(Pronunciation.path(root))
    made = Auxiliary.of(OptimConfig(semantic=0.5), root, CPU)
    assert isinstance(made, Joint)
    assert [type(t) for t in made.terms] == [SemanticTerm]
    both = Auxiliary.of(OptimConfig(semantic=0.5, phonetic=1.0), root, CPU)
    assert [type(t) for t in both.terms] == [SemanticTerm, PhoneticTerm]
    assert both.terms[1].weight == 1.0
    worded = Auxiliary.of(OptimConfig(semantic=0.5, lexical=2.0), root, CPU)
    assert [type(t) for t in worded.terms] == [SemanticTerm, LexicalTerm]
    spelled = Auxiliary.of(OptimConfig(phonetic=1.0, positions=2), root, CPU)
    assert cast("PhoneticTerm", spelled.terms[0]).positions == 2
