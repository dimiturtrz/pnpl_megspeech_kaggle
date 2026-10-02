from dataclasses import replace

import torch
from torch import nn

from megspeech.data.runs import CHANNELS
from megspeech.data.vocabulary import WORDS
from megspeech.model.onsets import ContextConfig
from megspeech.model.sentence import SentenceNet
from megspeech.model.wordnet import ConvConfig

CONTEXT = ContextConfig(width=16, heads=2, layers=1, conv=ConvConfig(hidden=8, dilations=(1, 2)))
SAMPLES = 50
GENERATOR = torch.Generator().manual_seed(0)


def test_sentence_net_forward() -> None:
    model = SentenceNet(nn.Identity(), CONTEXT).eval()
    slot = torch.tensor([[0, 1, 3], [2, 3, 3]])
    x = torch.randn(3, CHANNELS, SAMPLES, generator=GENERATOR)
    context, word = model.forward(x, torch.tensor([[0.0, 0.2, 0.0], [0.0, 0.0, 0.0]]), slot)
    assert context.shape == word.shape == (2, 3, WORDS)
    assert torch.isfinite(context[slot < len(x)]).all()


def test_padding_rows_change_nothing() -> None:
    model = SentenceNet(nn.Identity(), replace(CONTEXT, dropout=0.0, conv=replace(CONTEXT.conv, dropout=0.0))).train()
    x = torch.randn(3, CHANNELS, SAMPLES, generator=GENERATOR)
    onset = torch.tensor([[0.0, 0.2, 0.0], [0.0, 0.0, 0.0]])
    exact = model.forward(x, onset, torch.tensor([[0, 1, 3], [2, 3, 3]]))
    padding = 100 * torch.randn(2, CHANNELS, SAMPLES, generator=GENERATOR)
    padded = model.forward(torch.cat([x, padding]), onset, torch.tensor([[0, 1, 5], [2, 5, 5]]))
    mask = torch.tensor([[True, True, False], [True, False, False]])
    for a, b in zip(exact, padded, strict=True):
        assert torch.allclose(a[mask], b[mask], atol=1e-5)


def test_features() -> None:
    model = SentenceNet(nn.Identity(), CONTEXT).eval()
    slot = torch.tensor([[0, 1, 3], [2, 3, 3]])
    x = torch.randn(3, CHANNELS, SAMPLES, generator=GENERATOR)
    h, slots = model.features(x, torch.zeros(2, 3), slot)
    assert h.shape == (2, 3, CONTEXT.width)
    assert slots.shape == (2, 3, model.words.width)
    assert slots[slot == len(x)].eq(0).all()


def test_classify() -> None:
    model = SentenceNet(nn.Identity(), CONTEXT).eval()
    slot = torch.tensor([[0, 1, 3], [2, 3, 3]])
    x = torch.randn(3, CHANNELS, SAMPLES, generator=GENERATOR)
    onset = torch.tensor([[0.0, 0.2, 0.0], [0.0, 0.0, 0.0]])
    whole = model(x, onset, slot)
    split = model.classify(*model.features(x, onset, slot), slot < len(x))
    assert all(torch.equal(a, b) for a, b in zip(whole, split, strict=True))
