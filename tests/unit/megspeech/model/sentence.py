from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
import torch
from torch import nn

from megspeech.data.constants import CHANNELS
from megspeech.data.labels.vocabulary import VOCAB_SIZE
from megspeech.model.config import ContextConfig
from megspeech.model.context.span import SpanConformer
from megspeech.model.input.scaler import Scaler
from megspeech.model.input.subjects import SubjectLayer
from megspeech.model.sentence import SentenceNet
from megspeech.model.words.conv import ConvConfig, ConvWords

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
    assert context.shape == word.shape == (2, 3, VOCAB_SIZE)
    assert torch.isfinite(context[slot < len(x)]).all()


def test_embed() -> None:
    model = SentenceNet(nn.Identity(), CONTEXT).eval()
    x = torch.randn(3, CHANNELS, SAMPLES, generator=GENERATOR)
    subject = torch.zeros(len(x), dtype=torch.long)
    embedded = model.embed(x, subject)
    assert embedded.shape == (3, CONTEXT.conv.width)
    _, slots = model.features(x, torch.zeros(1, 3), torch.tensor([[0, 1, 2]]), subject)
    assert torch.allclose(slots[0], embedded, atol=1e-6)


def test_word_logits() -> None:
    model = SentenceNet(nn.Identity(), CONTEXT).eval()
    embeddings = torch.randn(3, CONTEXT.conv.width, generator=GENERATOR)
    assert torch.equal(model.word_logits(embeddings), model.words.head(embeddings))


def test_prepare() -> None:
    model = SentenceNet(nn.Identity(), CONTEXT).eval()
    x = torch.randn(3, CHANNELS, SAMPLES, generator=GENERATOR)
    subject = torch.zeros(len(x), dtype=torch.long)
    assert model.prepare(x, subject).shape == x.shape
    words = cast("ConvWords", model.words)
    assert torch.equal(words.pool(words.taps(model.prepare(x, subject))[-1]), model.embed(x, subject))


def test_encoder() -> None:
    model = SentenceNet(nn.Identity(), CONTEXT)
    encoder = {id(p) for p in model.encoder()}
    assert {id(p) for p in model.words.parameters()} <= encoder
    assert not encoder & {id(p) for p in [*model.head.parameters(), *model.context.parameters()]}


def test_adopt() -> None:
    model = SentenceNet(nn.Identity(), CONTEXT)
    other = SentenceNet(nn.Identity(), CONTEXT)
    head = [p.clone() for p in model.head.parameters()]
    model.adopt(other.state_dict())
    assert all(torch.equal(a, b) for a, b in zip(model.encoder(), other.encoder(), strict=True))
    assert all(torch.equal(a, b) for a, b in zip(model.head.parameters(), head, strict=True))
    with pytest.raises(KeyError, match="encoder"):
        model.adopt({})


def test_readouts() -> None:
    model = SentenceNet(nn.Identity(), CONTEXT)
    assert model.readouts() == (model.head, model.words.head)


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


def test_refine() -> None:
    model = SentenceNet(nn.Identity(), CONTEXT).eval()
    h = torch.randn(2, 3, CONTEXT.width, generator=GENERATOR)
    mask = torch.tensor([[True, True, False], [True, False, False]])
    refined = model.refine(h, mask)
    assert refined.shape == h.shape
    assert torch.equal(model.refine(h.where(mask[..., None], 7.0), mask)[mask], refined[mask])


LATENCY = replace(CONTEXT, latency=True, dropout=0.0, conv=replace(CONTEXT.conv, segments=4, dropout=0.0))


def test_latency_tags() -> None:
    """Latency tags are learned and saved; one token per word saves none, so earlier checkpoints still load."""
    assert SentenceNet(nn.Identity(), LATENCY).latency.shape == (4, CONTEXT.width)
    assert "latency" in SentenceNet(nn.Identity(), LATENCY).state_dict()
    assert "latency" not in SentenceNet(nn.Identity(), CONTEXT).state_dict()


def test_latency_tokens() -> None:
    """Each word enters the transformer as one token per chunk and leaves as one; padding words change nothing."""
    model = SentenceNet(nn.Identity(), LATENCY).train()
    x = torch.randn(3, CHANNELS, SAMPLES, generator=GENERATOR)
    onset = torch.tensor([[0.0, 0.2, 0.0], [0.0, 0.0, 0.0]])
    h, _ = model.features(x, onset, torch.tensor([[0, 1, 3], [2, 3, 3]]), torch.zeros(3, dtype=torch.long))
    assert h.shape == (2, 3 * 4, CONTEXT.width)
    exact = model.forward(x, onset, torch.tensor([[0, 1, 3], [2, 3, 3]]), torch.zeros(3, dtype=torch.long))
    padded = model.forward(
        torch.cat([x, torch.randn(2, CHANNELS, SAMPLES, generator=GENERATOR)]),
        onset,
        torch.tensor([[0, 1, 5], [2, 5, 5]]),
        torch.zeros(5, dtype=torch.long),
    )
    mask = torch.tensor([[True, True, False], [True, False, False]])
    assert exact[0].shape == (2, 3, VOCAB_SIZE)
    assert torch.allclose(exact[0][mask], padded[0][mask], atol=1e-5)


def test_whiten_saved_only_when_on() -> None:
    """A whitening model saves its matrix; one without saves none, so earlier checkpoints still load."""
    assert "whiten.matrix" in SentenceNet(nn.Identity(), replace(CONTEXT, whiten=True)).state_dict()
    assert not any(key.startswith("whiten") for key in SentenceNet(nn.Identity(), CONTEXT).state_dict())


def test_rhythm_adds_timing_logits() -> None:
    """With a timing head, the context logits gain its logits and the word head's stay; without, nothing saved."""
    model = SentenceNet(nn.Identity(), replace(CONTEXT, rhythm=True)).eval()
    x = torch.randn(3, CHANNELS, SAMPLES, generator=GENERATOR)
    onset = torch.tensor([[0.0, 0.2, 0.0], [0.0, 0.0, 0.0]])
    slot = torch.tensor([[0, 1, 3], [2, 3, 3]])
    subject = torch.zeros(len(x), dtype=torch.long)
    context, word = model.forward(x, onset, slot, subject)
    plain, same = model.classify(*model.features(x, onset, slot, subject), slot < len(x))
    assert torch.allclose(context - plain, model.timing(onset, slot < len(x)), atol=1e-5)
    assert torch.equal(word, same)
    assert not any(key.startswith("timing") for key in SentenceNet(nn.Identity(), CONTEXT).state_dict())


def test_heard() -> None:
    """The refined positions and slots the heads read, beside the same logits `forward` returns."""
    model = SentenceNet(nn.Identity(), CONTEXT).eval()
    x = torch.randn(3, CHANNELS, SAMPLES, generator=GENERATOR)
    onset = torch.tensor([[0.0, 0.2, 0.0], [0.0, 0.0, 0.0]])
    slot = torch.tensor([[0, 1, 3], [2, 3, 3]])
    refined, slots, context, word = model.heard(x, onset, slot, torch.zeros(len(x), dtype=torch.long))
    assert refined.shape == (2, 3, CONTEXT.width)
    assert slots.shape == (2, 3, model.words.width)
    whole = model(x, onset, slot, torch.zeros(len(x), dtype=torch.long))
    assert torch.equal(whole[0], context)
    assert torch.equal(whole[1], word)


def test_read() -> None:
    model = SentenceNet(nn.Identity(), CONTEXT).eval()
    refined = torch.randn(2, 3, CONTEXT.width, generator=GENERATOR)
    slots = torch.randn(2, 3, model.words.width, generator=GENERATOR)
    context, word = model.read(refined, slots)
    assert torch.equal(context, model.head(refined))
    assert torch.equal(word, model.words.head(slots))


def test_width() -> None:
    assert SentenceNet(nn.Identity(), CONTEXT).width == CONTEXT.width


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
    alone = SentenceNet.mixer(replace(CONTEXT, layers=0))
    h = torch.randn(2, 3, CONTEXT.width, generator=GENERATOR)
    assert torch.equal(alone(h, src_key_padding_mask=torch.zeros(2, 3, dtype=torch.bool)), h)
    assert not list(alone.parameters())
    assert cast("nn.TransformerEncoder", SentenceNet.mixer(CONTEXT)).norm is None
    assert isinstance(
        cast("nn.TransformerEncoder", SentenceNet.mixer(replace(CONTEXT, final_norm=True))).norm, nn.LayerNorm
    )


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


def test_load(tmp_path: Path) -> None:
    model = SentenceNet(Scaler(torch.randn(4, CHANNELS, SAMPLES, generator=GENERATOR)), CONTEXT)
    torch.save(model.state_dict(), tmp_path / "model.pt")
    CONTEXT.save(tmp_path)
    loaded = SentenceNet.load(tmp_path / "model.pt")
    assert all(torch.equal(loaded.state_dict()[key], value) for key, value in model.state_dict().items())


def test_subspace_channels(tmp_path: Path) -> None:
    """A model reading a `Projection`'s components: its input width is its scaler's, saved and loaded back so."""
    components = 5
    context = replace(CONTEXT, subjects=2, whiten=True)
    sample = torch.randn(4, components, SAMPLES, generator=GENERATOR)
    model = SentenceNet(Scaler(sample), context, channels=components).eval()
    model.whiten.fit(model.scale(sample))
    slot = torch.tensor([[0, 1, 3], [2, 3, 3]])
    x = torch.randn(3, components, SAMPLES, generator=GENERATOR)
    onsets = torch.tensor([[0.0, 0.2, 0.0], [0.0, 0.0, 0.0]])
    context_logits, _ = model.forward(x, onsets, slot, torch.zeros(len(x), dtype=torch.long))
    assert context_logits.shape == (2, 3, VOCAB_SIZE)
    torch.save(model.state_dict(), tmp_path / "model.pt")
    context.save(tmp_path)
    loaded = SentenceNet.load(tmp_path / "model.pt").eval()
    assert loaded.state_dict()["whiten.matrix"].shape == (components, components)
    again, _ = loaded.forward(x, onsets, slot, torch.zeros(len(x), dtype=torch.long))
    assert torch.allclose(again, context_logits)


def test_word_mask() -> None:
    """The mask enters before the timing is added, so a masked word still differs by position; with rate 0 there is
    no parameter and the state is the old one."""
    assert not any(key.startswith("masking") for key in SentenceNet(nn.Identity(), CONTEXT).state_dict())
    model = SentenceNet(nn.Identity(), replace(CONTEXT, word_mask=1.0)).train()
    assert any(key == "masking.token" for key in model.state_dict())
    x = torch.randn(3, CHANNELS, SAMPLES, generator=GENERATOR)
    slot = torch.tensor([[0, 1, 3], [2, 3, 3]])
    onset = torch.tensor([[0.0, 0.2, 0.0], [0.0, 0.0, 0.0]])
    h, _ = model.features(x, onset, slot, torch.zeros(len(x), dtype=torch.long))
    assert not torch.allclose(h[0, 0], h[0, 1])
    model.eval()
    h_eval, _ = model.features(x, onset, slot, torch.zeros(len(x), dtype=torch.long))
    assert not torch.allclose(h_eval[0, 0], h[0, 0])


def test_drop_path() -> None:
    model = SentenceNet(nn.Identity(), replace(CONTEXT, layers=2, drop_path=0.2))
    assert [layer.path.rate for layer in model.context.layers] == pytest.approx([0.1, 0.2])
