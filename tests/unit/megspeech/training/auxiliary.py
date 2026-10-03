from pathlib import Path
from typing import cast

import pytest
import torch
from torch import nn

from core.paths import DataRoot
from megspeech.data.constants import CHANNELS
from megspeech.data.labels.phonemes import Pronunciation
from megspeech.model.config import ContextConfig
from megspeech.model.pretraining import Pretraining
from megspeech.model.sentence import SentenceNet
from megspeech.model.words.conv import ConvConfig
from megspeech.objective.heads import DIM
from megspeech.objective.prototypes import Lexicon
from megspeech.training.auxiliary import Aligned, Auxiliary, Joint, Lexical, Pronounced, Silent
from megspeech.training.config import OptimConfig

CPU = torch.device("cpu")
SAMPLES = 50
GENERATOR = torch.Generator().manual_seed(0)
SLOT = torch.tensor([[0, 1, 3], [2, 3, 3]])
ONSET = torch.tensor([[0.0, 0.2, 0.0], [0.0, 0.0, 0.0]])


class Root:
    def __init__(self, path: Path) -> None:
        self.base = path

    def processed(self) -> Path:
        return self.base


def net() -> SentenceNet:
    return SentenceNet(
        nn.Identity(), ContextConfig(width=16, heads=2, layers=1, conv=ConvConfig(hidden=8, dilations=(1, 2)))
    )


def x() -> torch.Tensor:
    return torch.randn(3, CHANNELS, SAMPLES, generator=GENERATOR)


def aligned() -> Aligned:
    return Aligned(2.0, Lexicon(torch.tensor([10, 20]), torch.randn(2, DIM, generator=GENERATOR)))


def pronounced() -> Pronounced:
    return Pronounced(0.5, Pronunciation(torch.tensor([10, 20]), torch.tensor([[3, 1], [7, -1]])))


def identity(model: nn.Module) -> nn.Module:
    return model


TOKENS = torch.tensor([[10, 20, -100], [10, -100, -100]])


def test_silent_trained() -> None:
    model = net()
    assert Silent().trained(model, identity) is model


def test_joint_trained() -> None:
    model = net()
    trained = Joint((aligned(), pronounced())).trained(model, identity)
    assert isinstance(trained, Pretraining)
    assert trained.net is model
    assert len(trained.heads) == 2


def test_silent_loss() -> None:
    assert Silent().loss(net(), [], SLOT) == 0.0


def test_joint_loss() -> None:
    rule = Joint((aligned(), pronounced()))
    model = rule.trained(net(), identity)
    _, _, projected, phonetic = model(x(), ONSET, SLOT, torch.zeros(len(x()), dtype=torch.long))
    value = rule.loss(model, [projected, phonetic], TOKENS)
    heads = cast("Pretraining", model).heads
    parts = [
        term.loss(head, out, TOKENS) for term, head, out in zip(rule.terms, heads, [projected, phonetic], strict=True)
    ]
    assert value.item() == pytest.approx(sum(p.item() for p in parts))
    value.backward()
    assert all(p.grad is not None for p in heads.parameters())


def test_of(tmp_path: Path) -> None:
    root = cast("DataRoot", Root(tmp_path))
    assert isinstance(Auxiliary.of(OptimConfig(), root, CPU), Silent)
    with pytest.raises(FileNotFoundError):
        Auxiliary.of(OptimConfig(semantic=1.0), root, CPU)
    with pytest.raises(FileNotFoundError):
        Auxiliary.of(OptimConfig(phonetic=1.0), root, CPU)
    aligned().lexicon.save(Lexicon.path(root))
    pronounced().pronunciation.save(Pronunciation.path(root))
    made = Auxiliary.of(OptimConfig(semantic=0.5), root, CPU)
    assert isinstance(made, Joint)
    assert [type(t) for t in made.terms] == [Aligned]
    both = Auxiliary.of(OptimConfig(semantic=0.5, phonetic=1.0), root, CPU)
    assert isinstance(both, Joint)
    assert [type(t) for t in both.terms] == [Aligned, Pronounced]
    assert cast("Pronounced", both.terms[1]).weight == 1.0
    worded = Auxiliary.of(OptimConfig(semantic=0.5, lexical=2.0), root, CPU)
    assert [type(t) for t in cast("Joint", worded).terms] == [Aligned, Lexical]
    spelled = Auxiliary.of(OptimConfig(phonetic=1.0, positions=2), root, CPU)
    assert cast("Pronounced", cast("Joint", spelled).terms[0]).positions == 2
