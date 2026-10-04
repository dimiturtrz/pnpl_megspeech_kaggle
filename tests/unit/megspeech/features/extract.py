import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from core.paths import DataRoot
from megspeech.data.batching.split import HOLDOUT_FEATURES, TRAIN, Features, Split
from megspeech.data.batching.windows import Windows
from megspeech.data.cache.holdout import Track
from megspeech.data.config import DataConfig
from megspeech.data.corpus import Corpus
from megspeech.data.spans.spans import PAD
from megspeech.features.extract import (
    WORD,
    Averaged,
    Export,
    Extractor,
    Pooled,
    Projection,
    Read,
    Resolved,
    Segments,
    Windowed,
)
from megspeech.model.input.scaler import Scaler
from megspeech.model.words.trunk import DOWNSAMPLING, LATENT
from tests.unit.megspeech.model.words.megxl import words

MEG = torch.arange(5 * 2 * 8, dtype=torch.float32).reshape(5, 2, 8).sin()
COMPONENTS = 3
SPAN = 3
GAP = 20


def extractor(chunk: int = 5) -> Extractor:
    torch.manual_seed(0)
    encoder = words().eval()
    return Extractor(encoder, Scaler(MEG), Pooled(encoder), chunk)


def split(spans: int) -> Split:
    """`spans` spans of `SPAN` words `GAP` samples apart, each stopped where the next begins, plus one word in none."""
    count = spans * SPAN + 1
    at = torch.arange(count) * GAP
    end = (at // (SPAN * GAP) + 1) * SPAN * GAP
    meg = torch.randn(2, count * GAP + 64, generator=torch.Generator().manual_seed(0))
    index = torch.arange(spans * SPAN).view(spans, SPAN)
    zeros = torch.zeros(count, dtype=torch.long)
    return Split(Windows(meg, at, end), zeros, zeros, index, index.float(), torch.ones(count), zeros)


def segments(chunk: int = 2) -> Segments:
    return Segments(extractor(), lambda x: x[..., :COMPONENTS], length=SPAN * GAP, chunk=chunk)


def export(directory: Path) -> Export:
    corpus = Corpus(split(1), {"Sherlock1": split(2)}, frozenset())
    return Export(corpus, split(3), Features(directory))


def test_splits(tmp_path: Path):
    splits = export(tmp_path).splits()
    assert list(splits) == [TRAIN, "Sherlock1", f"{HOLDOUT_FEATURES}-{Track.DEEP}"]
    assert [len(s.index) for s in splits.values()] == [1, 2, 3]


def test_of_refuses_subspace(tmp_path: Path):
    with pytest.raises(ValueError, match="306 sensors"):
        Export.of(DataConfig(subspace=True), DataRoot(tmp_path), "name", torch.device("cpu"))


def test_save(tmp_path: Path):
    target = export(tmp_path)
    target.save(lambda s: torch.ones(len(s.windows), 2, 1), {"read": "pooled"})
    assert torch.load(tmp_path / "Sherlock1.pt").shape == (2 * SPAN + 1, 2, 1)
    assert json.loads((tmp_path / "config.json").read_text()) == {"read": "pooled"}


def test_call():
    windows = Windows.stacked(MEG)
    whole = extractor()(windows)
    assert whole.shape == (5, 2 * LATENT, 1)
    assert whole.dtype == torch.bfloat16
    assert torch.allclose(extractor(2)(windows).float(), whole.float())
    averaged = Averaged(extractor().words)
    assert averaged(MEG).shape == (5, 2, LATENT)
    resolved = Resolved(averaged, Projection.fit(averaged(MEG), COMPONENTS))
    assert resolved(MEG).shape == (5, 2, COMPONENTS)
    assert torch.equal(Windowed(extractor())(split(1)), extractor()(split(1).windows))


def test_scaled():
    base = extractor()
    assert torch.equal(base.scaled(MEG), base.scaler(MEG))


def test_tokens():
    base = extractor()
    torch.testing.assert_close(base.tokens(MEG), base.words.trunk(base.scaled(MEG)))
    assert base.tokens(MEG).shape[:2] == (5, 2)


def test_fit():
    sample = torch.randn(400, 2, LATENT, generator=torch.Generator().manual_seed(0))
    torch.manual_seed(0)
    fitted = Projection.fit(sample, COMPONENTS)
    projected = fitted(sample.flatten(0, 1))
    assert torch.allclose(projected.std(0), torch.ones(COMPONENTS), atol=0.05)
    assert torch.allclose(fitted.mean, sample.flatten(0, 1).mean(0))


def test_weights():
    at = torch.tensor([0, 30, 100, 7])
    weights = Segments.weights(at, torch.tensor([[0, 1, 2], [3, PAD, PAD]]), 12)
    for word, start in enumerate((0, 30, 100)):
        expected = torch.zeros(12)
        expected[start // DOWNSAMPLING : (start + WORD - 1) // DOWNSAMPLING + 1] = 1
        torch.testing.assert_close(weights[0, word], expected / expected.sum())
    torch.testing.assert_close(weights[1, 0], weights[0, 0])


def test_part():
    base = segments()
    data = split(2)
    segment = replace(data.windows, length=SPAN * GAP)
    index = torch.tensor([[0, 1, 2], [3, PAD, PAD]])
    words, features = base.part(segment, index)
    assert words.tolist() == [0, 1, 2, 3]
    assert features.shape == (4, 2, COMPONENTS)
    assert features.dtype == torch.bfloat16


def test_pooled():
    data = split(3)
    words, features = segments().pooled(data, torch.tensor([2, 0]))
    assert words.tolist() == [6, 7, 8, 0, 1, 2]
    torch.testing.assert_close(features, segments(1).pooled(data, torch.tensor([2, 0]))[1])


def test_segments_call():
    data = split(3)
    features = segments()(data)
    assert features.shape == (len(data.windows), 2, COMPONENTS)
    assert torch.equal(features[-1], torch.zeros(2, COMPONENTS, dtype=torch.bfloat16))
    assert features[:-1].abs().sum(dim=(1, 2)).gt(0).all()


def test_spanned():
    torch.manual_seed(0)
    sample = Read.spanned(extractor(), split(2))
    assert sample.shape == (2 * SPAN, 2, LATENT)


def test_extraction():
    base = extractor()
    sample = split(4)
    assert isinstance(Read.POOLED.extraction(base, sample), Windowed)
    resolved = Read.RESOLVED.extraction(base, sample)
    assert isinstance(resolved, Windowed)
    assert isinstance(resolved.extractor.read, Resolved)
    spanned = Read.SPANNED.extraction(base, sample)
    assert isinstance(spanned, Segments)
    assert spanned(sample).shape[1] == 2
