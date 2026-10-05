from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from core.paths import DataRoot
from megspeech.data.labels.context_tags import CLASSES, ContextTags
from megspeech.data.labels.phonemes import Pronunciation
from megspeech.data.labels.syntax import PartsOfSpeech
from megspeech.model.auxiliary import Tap, WithAuxiliary
from megspeech.model.config import ContextConfig
from megspeech.model.sentence import SentenceNet
from megspeech.model.words.conv import ConvConfig
from megspeech.model.words.heads import WordHead
from megspeech.objective.auxiliary.lexical import LexicalTerm
from megspeech.objective.auxiliary.phonetic import PhoneticTerm
from megspeech.objective.auxiliary.semantic import SemanticTerm
from megspeech.objective.auxiliary.speech import SpeechTerm
from megspeech.objective.auxiliary.syntactic import SyntacticTerm
from megspeech.objective.auxiliary.tagging import TagTerm
from megspeech.objective.scorer import DIM
from megspeech.objective.targets.speech import Targets
from megspeech.objective.targets.t5 import Lexicon
from megspeech.training.auxiliary import Auxiliary, Joint, NoAuxiliary
from megspeech.training.config import OptimConfig

CPU = torch.device("cpu")
GENERATOR = torch.Generator().manual_seed(0)
WORDS = pd.DataFrame({"run": ["a", "b"], "position": [1, 0]})


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
    tapped = Joint((semantic(),), Tap.CONTEXT).trained(model, identity)
    assert tapped.tap is Tap.CONTEXT
    assert tapped.heads[0](torch.zeros(1, model.width)).shape == (1, DIM)


def test_of(tmp_path: Path) -> None:
    root = cast("DataRoot", Root(tmp_path))
    assert isinstance(Auxiliary.of(OptimConfig(), root, CPU, WORDS), NoAuxiliary)
    with pytest.raises(FileNotFoundError):
        Auxiliary.of(OptimConfig(semantic=1.0), root, CPU, WORDS)
    with pytest.raises(FileNotFoundError):
        Auxiliary.of(OptimConfig(phonetic=1.0), root, CPU, WORDS)
    semantic().lexicon.save(Lexicon.path(root))
    phonetic().pronunciation.save(Pronunciation.path(root))
    made = Auxiliary.of(OptimConfig(semantic=0.5), root, CPU, WORDS)
    assert isinstance(made, Joint)
    assert [type(t) for t in made.terms] == [SemanticTerm]
    assert made.tap is Tap.WORD
    assert cast("Joint", Auxiliary.of(OptimConfig(semantic=0.5, tap=Tap.CONTEXT), root, CPU, WORDS)).tap is Tap.CONTEXT
    both = Auxiliary.of(OptimConfig(semantic=0.5, phonetic=1.0), root, CPU, WORDS)
    assert [type(t) for t in both.terms] == [SemanticTerm, PhoneticTerm]
    assert both.terms[1].weight == 1.0
    worded = Auxiliary.of(OptimConfig(semantic=0.5, lexical=2.0), root, CPU, WORDS)
    assert [type(t) for t in worded.terms] == [SemanticTerm, LexicalTerm]
    spelled = Auxiliary.of(OptimConfig(phonetic=1.0, positions=2), root, CPU, WORDS)
    assert cast("PhoneticTerm", spelled.terms[0]).positions == 2
    with pytest.raises(FileNotFoundError):
        Auxiliary.of(OptimConfig(syntactic=1.0), root, CPU, WORDS)
    PartsOfSpeech(torch.tensor([10, 20]), torch.tensor([7, 15])).save(PartsOfSpeech.path(root))
    tagged = Auxiliary.of(OptimConfig(phonetic=1.0, syntactic=0.3), root, CPU, WORDS)
    assert [type(t) for t in tagged.terms] == [PhoneticTerm, SyntacticTerm]
    with pytest.raises(FileNotFoundError):
        Auxiliary.of(OptimConfig(tagging=1.0), root, CPU, WORDS)
    ContextTags(np.array(["a", "b"]), np.array([0.5, 1.5]), np.array([3, 5])).save(ContextTags.path(root))
    tags = Auxiliary.of(OptimConfig(tagging=0.5), root, CPU, WORDS.assign(onset=[0.5, 9.0]))
    assert cast("TagTerm", tags.terms[0]).tags.tolist() == [3, -1]
    with pytest.raises(FileNotFoundError):
        Auxiliary.of(OptimConfig(speech=1.0), root, CPU, WORDS)
    Targets(np.ones((2, DIM), np.float16)).save(Targets.directory(root) / "a.npy")
    heard = Auxiliary.of(OptimConfig(speech=1.0, semantic=0.5), root, CPU, WORDS)
    assert [type(t) for t in heard.terms] == [SpeechTerm, SemanticTerm]
    assert cast("SpeechTerm", heard.terms[0]).speech.rows.tolist() == [0, -1]


def test_joint_trained_tagged() -> None:
    tagged = SentenceNet(
        nn.Identity(),
        ContextConfig(width=16, heads=2, layers=1, conv=ConvConfig(hidden=8, dilations=(1, 2), head=WordHead.TAGGED)),
    )
    trained = Joint((TagTerm(1.0, torch.tensor([3, -1])),)).trained(tagged, identity)
    assert trained.heads[0](torch.zeros(1, tagged.words.width)).shape == (1, len(CLASSES))
    assert {id(p) for p in trained.heads.parameters()} <= {id(p) for p in tagged.parameters()}
    with pytest.raises(ValueError, match="tagged"):
        Joint((TagTerm(1.0, torch.tensor([3, -1])),)).trained(net(), identity)
