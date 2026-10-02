from pathlib import Path
from typing import cast

import pytest
import torch
from torch import nn

from core.paths import DataRoot
from megspeech.data.runs import CHANNELS
from megspeech.model.onsets import ContextConfig
from megspeech.model.pretraining import Pretraining
from megspeech.model.sentence import SentenceNet
from megspeech.model.wordnet import ConvConfig
from megspeech.objective.heads import DIM
from megspeech.objective.prototypes import Lexicon
from megspeech.training.auxiliary import Aligned, Auxiliary, Silent

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


def identity(model: nn.Module) -> nn.Module:
    return model


def test_silent_trained() -> None:
    model = net()
    assert Silent().trained(model, identity) is model


def test_aligned_trained() -> None:
    model = net()
    trained = aligned().trained(model, identity)
    assert isinstance(trained, Pretraining)
    assert trained.net is model


def test_silent_loss() -> None:
    assert Silent().loss(net(), [], SLOT) == 0.0


def test_aligned_loss() -> None:
    rule = aligned()
    model = rule.trained(net(), identity)
    _, _, projected = model(x(), ONSET, SLOT)
    tokens = torch.tensor([[10, 20, -100], [10, -100, -100]])
    value = rule.loss(model, [projected], tokens)
    assert value.ndim == 0
    assert value.item() > 0
    value.backward()
    assert cast("Pretraining", model).semantic.project.weight.grad is not None
    assert rule.loss(model, [projected], torch.full((2, 3), 99)).item() == 0.0


def test_of(tmp_path: Path) -> None:
    root = cast("DataRoot", Root(tmp_path))
    assert isinstance(Auxiliary.of(0.0, root, CPU), Silent)
    with pytest.raises(FileNotFoundError):
        Auxiliary.of(1.0, root, CPU)
    aligned().lexicon.save(Lexicon.path(root))
    made = Auxiliary.of(0.5, root, CPU)
    assert isinstance(made, Aligned)
    assert made.weight == 0.5
