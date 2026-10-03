import torch
from torch import nn

from megspeech.data.constants import CHANNELS
from megspeech.model.auxiliary import WithAuxiliary
from megspeech.model.config import ContextConfig
from megspeech.model.sentence import SentenceNet
from megspeech.model.words.conv import ConvConfig
from megspeech.objective.auxiliary.semantic import Semantic
from megspeech.objective.scorer import DIM

SAMPLES = 50
GENERATOR = torch.Generator().manual_seed(0)
SLOT = torch.tensor([[0, 1, 3], [2, 3, 3]])
ONSET = torch.tensor([[0.0, 0.2, 0.0], [0.0, 0.0, 0.0]])


def net() -> SentenceNet:
    return SentenceNet(
        nn.Identity(), ContextConfig(width=16, heads=2, layers=1, conv=ConvConfig(hidden=8, dilations=(1, 2)))
    )


def x() -> torch.Tensor:
    return torch.randn(3, CHANNELS, SAMPLES, generator=GENERATOR)


def test_forward() -> None:
    model = net().eval()
    auxiliary = WithAuxiliary(model, nn.ModuleList([Semantic(model.words.width), nn.Linear(16, 5)])).eval()
    inputs = (x(), ONSET, SLOT, torch.zeros(3, dtype=torch.long))
    context, word, projected, phonetic = auxiliary.forward(*inputs)
    assert all(torch.equal(a, b) for a, b in zip((context, word), model(*inputs), strict=True))
    assert projected.shape == (2, 3, DIM)
    assert phonetic.shape == (2, 3, 5)
    assert len(WithAuxiliary(model, nn.ModuleList()).forward(*inputs)) == 2


def test_readouts() -> None:
    model = net()
    heads = nn.ModuleList([nn.Linear(16, 5), nn.Linear(16, 2)])
    assert WithAuxiliary(model, heads).readouts() == (model.head, model.words.head, heads[0], heads[1])
