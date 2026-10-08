import torch
from torch import nn

from megspeech.data.batching.split import SpanInput
from megspeech.data.constants import CHANNELS
from megspeech.model.fusion import Fusion
from megspeech.model.input.scaler import Scaler
from megspeech.model.sentence import SentenceNet
from tests.unit.megspeech.model.sentence import CONTEXT, GENERATOR, SAMPLES

torch.manual_seed(0)


def test_features() -> None:
    model = SentenceNet(nn.Identity(), CONTEXT).eval()
    slot = torch.tensor([[0, 1, 3], [2, 3, 3]])
    x = torch.randn(3, CHANNELS, SAMPLES, generator=GENERATOR)
    h, slots = model.features(SpanInput(x, torch.zeros(2, 3), slot, torch.zeros(len(x), dtype=torch.long)))
    assert h.shape == (2, 3, CONTEXT.width)
    assert slots.shape == (2, 3, model.words.width)
    assert slots[slot == len(x)].eq(0).all()


def test_width() -> None:
    assert SentenceNet(nn.Identity(), CONTEXT).width == CONTEXT.width


def test_fuse() -> None:
    net = SentenceNet(Scaler(torch.randn(4, CHANNELS, SAMPLES)), CONTEXT)
    assert net.fusion.names() == ("primary",)
    assert net.fuse(Fusion([], 0.3)) is net


def test_chunked() -> None:
    model = SentenceNet(nn.Identity(), CONTEXT).eval()
    slot = torch.tensor([[0, 1, 3], [2, 3, 3]])
    x = torch.randn(3, CHANNELS, SAMPLES, generator=GENERATOR)
    onset = torch.zeros(2, 3)
    subject = torch.zeros(len(x), dtype=torch.long)
    h, slots = model.features(SpanInput(x, onset, slot, subject))
    assert torch.equal(model.chunked(SpanInput(x, onset, slot, subject), slots), h)
