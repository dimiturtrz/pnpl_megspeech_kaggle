from dataclasses import replace

import torch
from torch import nn

from megspeech.data.batching.split import SpanInput
from megspeech.data.constants import CHANNELS
from megspeech.model.context.intervals import Intervals
from megspeech.model.encoder.fusion import Fusion
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


def test_centred() -> None:
    """Off, the embeddings themselves; on, each run's words centred on that run's mean."""
    slot = torch.tensor([[0, 1, 3], [2, 3, 3]])
    x = torch.randn(3, CHANNELS, SAMPLES, generator=GENERATOR)
    spans = SpanInput(x, torch.zeros(2, 3), slot, torch.zeros(len(x), dtype=torch.long), run=torch.tensor([0, 0, 1]))
    embedded = torch.randn(2, 3, 8, generator=GENERATOR) * spans.mask[..., None]
    assert SentenceNet(nn.Identity(), CONTEXT).centred(spans, embedded) is embedded
    on = SentenceNet(nn.Identity(), replace(CONTEXT, run_centred=True)).eval().centred(spans, embedded)
    assert torch.allclose(on[0, :2].mean(0), torch.zeros(8), atol=1e-5)
    assert torch.allclose(on[1, 0], torch.zeros(8), atol=1e-5)


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


def test_chunked_next_drop() -> None:
    """Evaluating, `next_drop` changes nothing; training, it reaches the timing input."""
    model = SentenceNet(nn.Identity(), replace(CONTEXT, intervals=Intervals.LOG, next_drop=1.0))
    whole = SentenceNet(nn.Identity(), replace(CONTEXT, intervals=Intervals.LOG))
    whole.load_state_dict(model.state_dict())
    slot = torch.tensor([[0, 1, 2]])
    x = torch.randn(3, CHANNELS, SAMPLES, generator=GENERATOR)
    spans = SpanInput(x, torch.tensor([[0.0, 0.3, 0.5]]), slot, torch.zeros(len(x), dtype=torch.long))
    slots = torch.randn(1, 3, model.words.width, generator=GENERATOR)
    assert torch.equal(model.eval().chunked(spans, slots), whole.eval().chunked(spans, slots))
    assert not torch.equal(model.train().chunked(spans, slots), whole.train().chunked(spans, slots))
