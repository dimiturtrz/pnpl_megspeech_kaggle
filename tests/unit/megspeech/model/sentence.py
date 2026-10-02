from dataclasses import replace
from typing import cast

import torch
from torch import nn

from megspeech.data.runs import CHANNELS
from megspeech.data.vocabulary import WORDS
from megspeech.model.onsets import ContextConfig
from megspeech.model.sentence import SentenceNet
from megspeech.model.span import SpanConformer
from megspeech.model.subjects import SubjectLayer
from megspeech.model.wordnet import ConvConfig

CONTEXT = ContextConfig(width=16, heads=2, layers=1, conv=ConvConfig(hidden=8, dilations=(1, 2)))
SAMPLES = 50
GENERATOR = torch.Generator().manual_seed(0)


def test_sentence_net_forward() -> None:
    model = SentenceNet(nn.Identity(), CONTEXT).eval()
    slot = torch.tensor([[0, 1, 3], [2, 3, 3]])
    x = torch.randn(3, CHANNELS, SAMPLES, generator=GENERATOR)
    context, word = model.forward(
        x, torch.tensor([[0.0, 0.2, 0.0], [0.0, 0.0, 0.0]]), slot, torch.zeros(len(x), dtype=torch.long)
    )
    assert context.shape == word.shape == (2, 3, WORDS)
    assert torch.isfinite(context[slot < len(x)]).all()


def test_padding_rows_change_nothing() -> None:
    model = SentenceNet(nn.Identity(), replace(CONTEXT, dropout=0.0, conv=replace(CONTEXT.conv, dropout=0.0))).train()
    x = torch.randn(3, CHANNELS, SAMPLES, generator=GENERATOR)
    onset = torch.tensor([[0.0, 0.2, 0.0], [0.0, 0.0, 0.0]])
    exact = model.forward(x, onset, torch.tensor([[0, 1, 3], [2, 3, 3]]), torch.zeros(3, dtype=torch.long))
    padding = 100 * torch.randn(2, CHANNELS, SAMPLES, generator=GENERATOR)
    padded = model.forward(
        torch.cat([x, padding]), onset, torch.tensor([[0, 1, 5], [2, 5, 5]]), torch.zeros(5, dtype=torch.long)
    )
    mask = torch.tensor([[True, True, False], [True, False, False]])
    for a, b in zip(exact, padded, strict=True):
        assert torch.allclose(a[mask], b[mask], atol=1e-5)


def test_features() -> None:
    model = SentenceNet(nn.Identity(), CONTEXT).eval()
    slot = torch.tensor([[0, 1, 3], [2, 3, 3]])
    x = torch.randn(3, CHANNELS, SAMPLES, generator=GENERATOR)
    h, slots = model.features(x, torch.zeros(2, 3), slot, torch.zeros(len(x), dtype=torch.long))
    assert h.shape == (2, 3, CONTEXT.width)
    assert slots.shape == (2, 3, model.words.width)
    assert slots[slot == len(x)].eq(0).all()


def test_features_instance_ignores_window_offset_and_gain() -> None:
    model = SentenceNet(nn.Identity(), replace(CONTEXT, conv=replace(CONTEXT.conv, instance=True))).eval()
    slot = torch.tensor([[0, 1, 3], [2, 3, 3]])
    x = torch.randn(3, CHANNELS, SAMPLES, generator=GENERATOR)
    x[:, :, 40:] = 0
    shifted = x.clone()
    shifted[:, :, :40] = 5.0 + 3.0 * x[:, :, :40]
    h, _ = model.features(x, torch.zeros(2, 3), slot, torch.zeros(len(x), dtype=torch.long))
    moved, _ = model.features(shifted, torch.zeros(2, 3), slot, torch.zeros(len(x), dtype=torch.long))
    assert torch.allclose(h, moved, atol=1e-4)


def test_classify() -> None:
    model = SentenceNet(nn.Identity(), CONTEXT).eval()
    slot = torch.tensor([[0, 1, 3], [2, 3, 3]])
    x = torch.randn(3, CHANNELS, SAMPLES, generator=GENERATOR)
    onset = torch.tensor([[0.0, 0.2, 0.0], [0.0, 0.0, 0.0]])
    whole = model(x, onset, slot, torch.zeros(len(x), dtype=torch.long))
    split = model.classify(*model.features(x, onset, slot, torch.zeros(len(x), dtype=torch.long)), slot < len(x))
    assert all(torch.equal(a, b) for a, b in zip(whole, split, strict=True))


def test_features_read_each_subject_through_its_layer() -> None:
    model = SentenceNet(nn.Identity(), replace(CONTEXT, subjects=2)).eval()
    slot = torch.tensor([[0, 1, 3], [2, 3, 3]])
    x = torch.randn(3, CHANNELS, SAMPLES, generator=GENERATOR)
    with torch.no_grad():
        cast("SubjectLayer", model.subjects).weight[1].mul_(-1.0)
    deep, _ = model.features(x, torch.zeros(2, 3), slot, torch.zeros(3, dtype=torch.long))
    other, _ = model.features(x, torch.zeros(2, 3), slot, torch.tensor([0, 0, 1]))
    assert torch.allclose(deep[0], other[0], atol=1e-6)
    assert not torch.allclose(deep[1, 0], other[1, 0], atol=1e-3)


def test_mixer() -> None:
    assert isinstance(SentenceNet.mixer(CONTEXT), nn.TransformerEncoder)
    assert isinstance(SentenceNet.mixer(replace(CONTEXT, mixing=3)), SpanConformer)


def test_padding_rows_change_nothing_mixing() -> None:
    mixing = replace(CONTEXT, dropout=0.0, mixing=3, conv=replace(CONTEXT.conv, dropout=0.0))
    model = SentenceNet(nn.Identity(), mixing).train()
    x = torch.randn(3, CHANNELS, SAMPLES, generator=GENERATOR)
    onset = torch.tensor([[0.0, 0.2, 0.0], [0.0, 0.0, 0.0]])
    exact = model.forward(x, onset, torch.tensor([[0, 1, 3], [2, 3, 3]]), torch.zeros(3, dtype=torch.long))
    padded = model.forward(
        torch.cat([x, torch.randn(2, CHANNELS, SAMPLES, generator=GENERATOR)]),
        onset,
        torch.tensor([[0, 1, 5], [2, 5, 5]]),
        torch.zeros(5, dtype=torch.long),
    )
    mask = torch.tensor([[True, True, False], [True, False, False]])
    assert torch.allclose(exact[0][mask], padded[0][mask], atol=1e-5)
