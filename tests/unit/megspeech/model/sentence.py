from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
import torch
from torch import nn

from core.paths import DataRoot
from megspeech.data.batching.split import SpanInput
from megspeech.data.constants import CHANNELS
from megspeech.data.labels.vocabulary import VOCAB_SIZE
from megspeech.model.config import ContextConfig, Encoder
from megspeech.model.context.droplayer import DropPathLayer
from megspeech.model.input.subjects import SubjectLayer
from megspeech.model.pyramid import Pyramid, PyramidConfig
from megspeech.model.sentence import SentenceNet
from megspeech.model.words.conv import ConvConfig, ConvWords
from megspeech.model.words.spanned import SpannedConfig, SpanWords
from megspeech.model.words.trunk import DOWNSAMPLING, Geometry, Trunk
from tests.unit.megspeech.model.words.trunk import CHANNELS as GEOMETRY
from tests.unit.megspeech.model.words.trunk import Features

CONTEXT = ContextConfig(width=16, heads=2, layers=1, conv=ConvConfig(hidden=8, dilations=(1, 2)))
SAMPLES = 50
GENERATOR = torch.Generator().manual_seed(0)


def test_sentence_net_forward() -> None:
    model = SentenceNet(nn.Identity(), CONTEXT).eval()
    slot = torch.tensor([[0, 1, 3], [2, 3, 3]])
    x = torch.randn(3, CHANNELS, SAMPLES, generator=GENERATOR)
    context, word = model.forward(
        SpanInput(x, torch.tensor([[0.0, 0.2, 0.0], [0.0, 0.0, 0.0]]), slot, torch.zeros(len(x), dtype=torch.long))
    )
    assert context.shape == word.shape == (2, 3, VOCAB_SIZE)
    assert torch.isfinite(context[slot < len(x)]).all()


def test_embed() -> None:
    model = SentenceNet(nn.Identity(), CONTEXT).eval()
    x = torch.randn(3, CHANNELS, SAMPLES, generator=GENERATOR)
    subject = torch.zeros(len(x), dtype=torch.long)
    embedded = model.embed(x, subject)
    assert embedded.shape == (3, CONTEXT.conv.width)
    _, slots = model.features(SpanInput(x, torch.zeros(1, 3), torch.tensor([[0, 1, 2]]), subject))
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


def spanned(monkeypatch: pytest.MonkeyPatch) -> SentenceNet:
    trunk = Trunk(Features(), Geometry.of(GEOMETRY))
    monkeypatch.setattr(Trunk, "pretrained", classmethod(lambda _cls, _root: trunk))
    monkeypatch.setattr(DataRoot, "from_config", classmethod(lambda _cls: DataRoot(Path())))
    config = replace(CONTEXT, encoder=Encoder.SPANNED, spanned=SpannedConfig(components=3, width=8))
    return SentenceNet(nn.Identity(), config, channels=2)


def test_encoder_spanned(monkeypatch: pytest.MonkeyPatch) -> None:
    model = spanned(monkeypatch)
    ids = [id(p) for p in model.encoder()]
    assert len(ids) == len(set(ids))
    assert {id(p) for p in model.words.backbone()} <= set(ids)
    words = model.words
    assert isinstance(words, SpanWords)
    assert not set(ids) & {id(p) for p in [*words.reduce.parameters(), *words.mix.parameters()]}


def test_features_spanned(monkeypatch: pytest.MonkeyPatch) -> None:
    model = spanned(monkeypatch).eval()
    x = torch.randn(2, 2, 3 * DOWNSAMPLING, generator=GENERATOR)
    onset = torch.tensor([[0.0, 0.3, 0.0], [0.0, 0.0, 0.0]])
    slot = torch.tensor([[0, 0, 2], [1, 2, 2]])
    subject = torch.zeros(2, dtype=torch.long)
    h, slots = model.features(SpanInput(x, onset, slot, subject))
    assert h.shape == (2, 3, CONTEXT.width)
    assert slots.shape == (2, 3, 8)
    assert torch.all(slots[slot == 2] == 0)
    context, word = model.forward(SpanInput(x, onset, slot, subject))
    assert context.shape == word.shape == (2, 3, VOCAB_SIZE)
    context[slot < 2].sum().backward()
    assert all(p.grad is not None for p in model.words.backbone())


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
    exact = model.forward(SpanInput(x, onset, torch.tensor([[0, 1, 3], [2, 3, 3]]), torch.zeros(3, dtype=torch.long)))
    padding = 100 * torch.randn(2, CHANNELS, SAMPLES, generator=GENERATOR)
    padded = model.forward(
        SpanInput(
            torch.cat([x, padding]), onset, torch.tensor([[0, 1, 5], [2, 5, 5]]), torch.zeros(5, dtype=torch.long)
        )
    )
    mask = torch.tensor([[True, True, False], [True, False, False]])
    for a, b in zip(exact, padded, strict=True):
        assert torch.allclose(a[mask], b[mask], atol=1e-5)


def test_features_instance_ignores_window_offset_and_gain() -> None:
    model = SentenceNet(nn.Identity(), replace(CONTEXT, conv=replace(CONTEXT.conv, instance=True))).eval()
    slot = torch.tensor([[0, 1, 3], [2, 3, 3]])
    x = torch.randn(3, CHANNELS, SAMPLES, generator=GENERATOR)
    x[:, :, 40:] = 0
    shifted = x.clone()
    shifted[:, :, :40] = 5.0 + 3.0 * x[:, :, :40]
    h, _ = model.features(SpanInput(x, torch.zeros(2, 3), slot, torch.zeros(len(x), dtype=torch.long)))
    moved, _ = model.features(SpanInput(shifted, torch.zeros(2, 3), slot, torch.zeros(len(x), dtype=torch.long)))
    assert torch.allclose(h, moved, atol=1e-4)


def test_classify() -> None:
    model = SentenceNet(nn.Identity(), CONTEXT).eval()
    slot = torch.tensor([[0, 1, 3], [2, 3, 3]])
    x = torch.randn(3, CHANNELS, SAMPLES, generator=GENERATOR)
    onset = torch.tensor([[0.0, 0.2, 0.0], [0.0, 0.0, 0.0]])
    whole = model(SpanInput(x, onset, slot, torch.zeros(len(x), dtype=torch.long)))
    split = model.classify(
        *model.features(SpanInput(x, onset, slot, torch.zeros(len(x), dtype=torch.long))), slot < len(x)
    )
    assert all(torch.equal(a, b) for a, b in zip(whole, split, strict=True))


def test_refine() -> None:
    model = SentenceNet(nn.Identity(), CONTEXT).eval()
    h = torch.randn(2, 3, CONTEXT.width, generator=GENERATOR)
    mask = torch.tensor([[True, True, False], [True, False, False]])
    refined = model.refine(h, mask)
    assert refined.shape == h.shape
    assert torch.equal(model.refine(h.where(mask[..., None], 7.0), mask)[mask], refined[mask])


UNDROPPED = replace(CONTEXT, dropout=0.0, conv=replace(CONTEXT.conv, segments=4, dropout=0.0))


def test_padding_words_change_nothing() -> None:
    """Each word enters the transformer as one token; padding words change nothing."""
    model = SentenceNet(nn.Identity(), UNDROPPED).train()
    x = torch.randn(3, CHANNELS, SAMPLES, generator=GENERATOR)
    onset = torch.tensor([[0.0, 0.2, 0.0], [0.0, 0.0, 0.0]])
    h, _ = model.features(SpanInput(x, onset, torch.tensor([[0, 1, 3], [2, 3, 3]]), torch.zeros(3, dtype=torch.long)))
    assert h.shape == (2, 3, CONTEXT.width)
    exact = model.forward(SpanInput(x, onset, torch.tensor([[0, 1, 3], [2, 3, 3]]), torch.zeros(3, dtype=torch.long)))
    padded = model.forward(
        SpanInput(
            torch.cat([x, torch.randn(2, CHANNELS, SAMPLES, generator=GENERATOR)]),
            onset,
            torch.tensor([[0, 1, 5], [2, 5, 5]]),
            torch.zeros(5, dtype=torch.long),
        )
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
    context, word = model.forward(SpanInput(x, onset, slot, subject))
    plain, same = model.classify(*model.features(SpanInput(x, onset, slot, subject)), slot < len(x))
    assert torch.allclose(context - plain, model.timing(onset, slot < len(x)), atol=1e-5)
    assert torch.equal(word, same)
    assert not any(key.startswith("timing") for key in SentenceNet(nn.Identity(), CONTEXT).state_dict())


def test_heard() -> None:
    """The refined positions and slots the heads read, beside the same logits `forward` returns."""
    model = SentenceNet(nn.Identity(), CONTEXT).eval()
    x = torch.randn(3, CHANNELS, SAMPLES, generator=GENERATOR)
    onset = torch.tensor([[0.0, 0.2, 0.0], [0.0, 0.0, 0.0]])
    slot = torch.tensor([[0, 1, 3], [2, 3, 3]])
    refined, slots, context, word = model.heard(SpanInput(x, onset, slot, torch.zeros(len(x), dtype=torch.long)))
    assert refined.shape == (2, 3, CONTEXT.width)
    assert slots.shape == (2, 3, model.words.width)
    whole = model(SpanInput(x, onset, slot, torch.zeros(len(x), dtype=torch.long)))
    assert torch.equal(whole[0], context)
    assert torch.equal(whole[1], word)


def test_read() -> None:
    model = SentenceNet(nn.Identity(), CONTEXT).eval()
    refined = torch.randn(2, 3, CONTEXT.width, generator=GENERATOR)
    slots = torch.randn(2, 3, model.words.width, generator=GENERATOR)
    context, word = model.read(refined, slots)
    assert torch.equal(context, model.head(refined))
    assert torch.equal(word, model.words.head(slots))


def test_features_read_each_subject_through_its_layer() -> None:
    model = SentenceNet(nn.Identity(), replace(CONTEXT, subjects=2)).eval()
    slot = torch.tensor([[0, 1, 3], [2, 3, 3]])
    x = torch.randn(3, CHANNELS, SAMPLES, generator=GENERATOR)
    with torch.no_grad():
        cast("SubjectLayer", model.subjects).weight[1].mul_(-1.0)
    deep, _ = model.features(SpanInput(x, torch.zeros(2, 3), slot, torch.zeros(3, dtype=torch.long)))
    other, _ = model.features(SpanInput(x, torch.zeros(2, 3), slot, torch.tensor([0, 0, 1])))
    assert torch.allclose(deep[0], other[0], atol=1e-6)
    assert not torch.allclose(deep[1, 0], other[1, 0], atol=1e-3)


def test_padding_rows_change_nothing_mixing() -> None:
    mixing = replace(CONTEXT, dropout=0.0, mixing=3, conv=replace(CONTEXT.conv, dropout=0.0))
    model = SentenceNet(nn.Identity(), mixing).train()
    x = torch.randn(3, CHANNELS, SAMPLES, generator=GENERATOR)
    onset = torch.tensor([[0.0, 0.2, 0.0], [0.0, 0.0, 0.0]])
    exact = model.forward(SpanInput(x, onset, torch.tensor([[0, 1, 3], [2, 3, 3]]), torch.zeros(3, dtype=torch.long)))
    padded = model.forward(
        SpanInput(
            torch.cat([x, torch.randn(2, CHANNELS, SAMPLES, generator=GENERATOR)]),
            onset,
            torch.tensor([[0, 1, 5], [2, 5, 5]]),
            torch.zeros(5, dtype=torch.long),
        )
    )
    mask = torch.tensor([[True, True, False], [True, False, False]])
    assert torch.allclose(exact[0][mask], padded[0][mask], atol=1e-5)


def test_word_mask() -> None:
    """The mask enters before the timing is added, so a masked word still differs by position; with rate 0 there is
    no parameter and the state is the old one."""
    assert not any(key.startswith("masking") for key in SentenceNet(nn.Identity(), CONTEXT).state_dict())
    model = SentenceNet(nn.Identity(), replace(CONTEXT, word_mask=1.0)).train()
    assert any(key == "masking.token" for key in model.state_dict())
    x = torch.randn(3, CHANNELS, SAMPLES, generator=GENERATOR)
    slot = torch.tensor([[0, 1, 3], [2, 3, 3]])
    onset = torch.tensor([[0.0, 0.2, 0.0], [0.0, 0.0, 0.0]])
    h, _ = model.features(SpanInput(x, onset, slot, torch.zeros(len(x), dtype=torch.long)))
    assert not torch.allclose(h[0, 0], h[0, 1])
    model.eval()
    h_eval, _ = model.features(SpanInput(x, onset, slot, torch.zeros(len(x), dtype=torch.long)))
    assert not torch.allclose(h_eval[0, 0], h[0, 0])


def test_drop_path() -> None:
    model = SentenceNet(nn.Identity(), replace(CONTEXT, layers=2, drop_path=0.2))
    assert [
        cast("DropPathLayer", layer).path.rate for layer in cast("nn.TransformerEncoder", model.context).layers
    ] == pytest.approx([0.1, 0.2])


def test_heard_timed() -> None:
    """`heard`'s outputs exactly, and the encoder's time course at each position, zeros where it is empty."""
    model = SentenceNet(nn.Identity(), CONTEXT).eval()
    x = torch.randn(3, CHANNELS, SAMPLES, generator=GENERATOR)
    onset = torch.tensor([[0.0, 0.2, 0.0], [0.0, 0.0, 0.0]])
    slot = torch.tensor([[0, 1, 3], [2, 3, 3]])
    subject = torch.zeros(len(x), dtype=torch.long)
    *heard, times = model.heard_timed(SpanInput(x, onset, slot, subject))
    assert all(torch.equal(a, b) for a, b in zip(heard, model.heard(SpanInput(x, onset, slot, subject)), strict=True))
    assert times.shape == (2, 3, model.words.time_width(), SAMPLES)
    assert times[slot == len(x)].eq(0).all()


def test_finish() -> None:
    model = SentenceNet(nn.Identity(), CONTEXT).eval()
    slot = torch.tensor([[0, 1, 3], [2, 3, 3]])
    x = torch.randn(3, CHANNELS, SAMPLES, generator=GENERATOR)
    onset = torch.zeros(2, 3)
    subject = torch.zeros(len(x), dtype=torch.long)
    h, slots = model.features(SpanInput(x, onset, slot, subject))
    finished = model.finish(h, slots, onset, slot < len(x))
    assert all(
        torch.equal(a, b) for a, b in zip(finished, model.heard(SpanInput(x, onset, slot, subject)), strict=True)
    )


TAPPED = replace(CONTEXT, conv=replace(CONTEXT.conv, pyramid=PyramidConfig(taps=(0, 1), width=6)))


def spans() -> SpanInput:
    x = torch.randn(3, CHANNELS, SAMPLES, generator=GENERATOR)
    return SpanInput(
        x,
        torch.tensor([[0.0, 0.2, 0.0], [0.0, 0.0, 0.0]]),
        torch.tensor([[0, 1, 3], [2, 3, 3]]),
        torch.zeros(len(x), dtype=torch.long),
    )


def test_train() -> None:
    """A pyramid model holds everything under the readout frozen in evaluation mode, whatever mode it is set to;
    without one every module follows the mode."""
    plain = SentenceNet(nn.Identity(), CONTEXT).train()
    assert plain.context.training
    assert all(p.requires_grad for p in plain.parameters())
    tapped = SentenceNet(nn.Identity(), TAPPED)
    assert tapped.train() is tapped
    assert isinstance(tapped.pyramid, Pyramid)
    assert tapped.pyramid.training
    assert not tapped.words.training
    assert not tapped.context.training
    trainable = {name.split(".")[0] for name, p in tapped.named_parameters() if p.requires_grad}
    assert trainable == {"pyramid"}


def test_warm() -> None:
    """A pyramid model warms from the plain model it reads, logits unchanged until its readout trains; any other
    missing or unknown entry is refused."""
    plain = SentenceNet(nn.Identity(), CONTEXT).eval()
    tapped = SentenceNet(nn.Identity(), TAPPED)
    tapped.warm(plain.state_dict())
    tapped.eval()
    batch = spans()
    for ours, theirs in zip(tapped(batch), plain(batch), strict=True):
        assert torch.allclose(ours, theirs, atol=1e-6)
    with pytest.raises(KeyError, match="lacks"):
        plain.warm({k: v for k, v in plain.state_dict().items() if not k.startswith("head.")})
    with pytest.raises(KeyError, match="unknown"):
        plain.warm(tapped.state_dict())


def test_pyramid_trains_only_its_readout() -> None:
    tapped = SentenceNet(nn.Identity(), TAPPED)
    context, word = tapped(spans())
    (context.sum() + word.sum()).backward()
    assert all(p.grad is None for name, p in tapped.named_parameters() if not name.startswith("pyramid."))
    grad = cast("Pyramid", tapped.pyramid).out.weight.grad
    assert grad is not None
    assert grad.abs().sum() > 0
