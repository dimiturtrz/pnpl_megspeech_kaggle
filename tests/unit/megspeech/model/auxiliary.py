import torch
from torch import nn

from megspeech.data.constants import CHANNELS
from megspeech.model.auxiliary import Tap, WithAuxiliary
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
    tapped = WithAuxiliary(model, nn.ModuleList([nn.Identity()]), Tap.CONTEXT).eval().forward(*inputs)
    h, _ = model.features(*inputs)
    assert torch.equal(tapped[2], model.refine(h, SLOT < 3))


def test_forward_time() -> None:
    """A head on `Tap.TIME` reads the window's time course, and the model's own outputs are those of `forward`
    without it."""
    model = net().eval()
    head = nn.Conv1d(model.words.time_width(), 1, 1)
    inputs = (x(), ONSET, SLOT, torch.zeros(3, dtype=torch.long))
    timed = WithAuxiliary(model, nn.ModuleList([nn.Identity()]), taps=(Tap.TIME,)).eval().forward(*inputs)
    plain = WithAuxiliary(model, nn.ModuleList()).eval().forward(*inputs)
    assert all(torch.equal(a, b) for a, b in zip(plain, timed[:2], strict=True))
    assert timed[2].shape == (2, 3, model.words.time_width(), SAMPLES)
    assert head(timed[2].flatten(0, 1)).shape == (6, 1, SAMPLES)


def test_width() -> None:
    model = net()
    assert [tap.width(model) for tap in Tap.pooled()] == [model.words.width, 16]
    assert Tap.TIME.width(model) == model.words.time_width()


def test_pooled() -> None:
    assert Tap.pooled() == (Tap.WORD, Tap.CONTEXT)


def test_pick() -> None:
    slots, refined = torch.zeros(2, 3, 4), torch.ones(2, 3, 16)
    times = torch.full((2, 3, 8, 5), 2.0)
    assert Tap.TIME.pick(slots, refined, times) is times
    assert Tap.WORD.pick(slots, refined) is slots
    assert Tap.CONTEXT.pick(slots, refined) is refined


def test_readouts() -> None:
    model = net()
    heads = nn.ModuleList([nn.Linear(16, 5), nn.Linear(16, 2)])
    assert WithAuxiliary(model, heads).readouts() == (model.head, model.words.head, heads[0], heads[1])
