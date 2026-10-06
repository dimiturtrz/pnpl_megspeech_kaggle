from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import Tensor

from core.paths import DataRoot
from megspeech.data.labels.vocabulary import IGNORE, VOCAB_SIZE
from megspeech.objective.anchors import Anchors, Source
from megspeech.objective.targets.speech import Targets
from megspeech.objective.targets.t5 import Prototypes

DIM = 4
GENERATOR = torch.Generator().manual_seed(0)
ROWS = 2 * VOCAB_SIZE
MISSING_ROW = VOCAB_SIZE


def labels() -> Tensor:
    return torch.arange(ROWS) % VOCAB_SIZE


def words() -> pd.DataFrame:
    return pd.DataFrame({"run": ["r0"] * ROWS, "position": np.arange(ROWS)})


def write_targets(root: DataRoot) -> None:
    vectors = np.repeat(np.arange(ROWS, dtype=np.float16)[:, None], DIM, axis=1)
    vectors[MISSING_ROW] = np.nan
    Targets(vectors).save(Targets.directory(root) / "r0.npy")


@pytest.mark.parametrize(("source", "active"), [(Source.NONE, False), (Source.AUDIO, True), (Source.TEXT, True)])
def test_active(source: Source, *, active: bool) -> None:
    assert source.active is active


def test_audio(tmp_path: Path) -> None:
    root = DataRoot(tmp_path)
    write_targets(root)
    anchors = Anchors.audio(root, words(), labels())
    assert anchors.vectors.shape == (VOCAB_SIZE, DIM)
    assert anchors.counts[0] == 1
    assert (anchors.counts[1:] == 2).all()
    assert torch.allclose(anchors.vectors[0], torch.zeros(DIM))
    assert torch.allclose(anchors.vectors[7], torch.full((DIM,), 7 + VOCAB_SIZE / 2))


def test_audio_drops_unlabelled_rows(tmp_path: Path) -> None:
    root = DataRoot(tmp_path)
    write_targets(root)
    ignored = labels().clone()
    ignored[VOCAB_SIZE + 3] = IGNORE
    anchors = Anchors.audio(root, words(), ignored)
    assert anchors.counts[3] == 1
    assert torch.allclose(anchors.vectors[3], torch.full((DIM,), 3.0))


def test_audio_rejects_a_word_without_targets(tmp_path: Path) -> None:
    root = DataRoot(tmp_path)
    write_targets(root)
    with pytest.raises(ValueError, match="no speech target"):
        Anchors.audio(root, words(), labels().clamp(max=VOCAB_SIZE - 2))
    with pytest.raises(ValueError, match="labels"):
        Anchors.audio(root, words(), labels()[:-1])


def test_text(tmp_path: Path) -> None:
    root = DataRoot(tmp_path)
    vectors = torch.randn(VOCAB_SIZE, DIM, generator=GENERATOR)
    Prototypes(vectors).save(Prototypes.path(root))
    anchors = Anchors.text(root, words(), labels())
    assert torch.equal(anchors.vectors, vectors)
    assert (anchors.counts == 0).all()


def test_key() -> None:
    assert Anchors.key(words(), labels()) == Anchors.key(words(), labels())
    assert Anchors.key(words(), labels()) != Anchors.key(words().iloc[::-1].reset_index(drop=True), labels())
    assert Anchors.key(words(), labels()) != Anchors.key(words(), labels().flip(0))


def test_path(tmp_path: Path) -> None:
    root = DataRoot(tmp_path)
    assert Anchors.path(root, Source.AUDIO, "k").name == "audio-k.npz"
    assert Anchors.path(root, Source.AUDIO, "k").parent == root.processed() / "anchors"


def test_save(tmp_path: Path) -> None:
    anchors = Anchors(torch.randn(VOCAB_SIZE, DIM, generator=GENERATOR), torch.arange(VOCAB_SIZE))
    assert anchors.save(tmp_path / "nested" / "a.npz").exists()


def test_load(tmp_path: Path) -> None:
    anchors = Anchors(torch.randn(VOCAB_SIZE, DIM, generator=GENERATOR), torch.arange(VOCAB_SIZE))
    loaded = Anchors.load(anchors.save(tmp_path / "a.npz"))
    assert torch.equal(loaded.vectors, anchors.vectors)
    assert torch.equal(loaded.counts, anchors.counts)


def spanning() -> Anchors:
    return Anchors(torch.randn(VOCAB_SIZE, 64, generator=GENERATOR), torch.arange(VOCAB_SIZE))


def test_spectrum() -> None:
    spectrum = spanning().spectrum()
    assert (spectrum.diff() >= 0).all()
    assert spectrum[-1].item() == pytest.approx(1.0)
    assert spectrum[VOCAB_SIZE - 2].item() == pytest.approx(1.0, abs=1e-5)


def test_reduce() -> None:
    anchors = spanning()
    assert anchors.reduce(0) is anchors
    reduced = anchors.reduce(8)
    assert reduced.vectors.shape == (VOCAB_SIZE, 8)
    assert torch.equal(reduced.counts, anchors.counts)
    assert torch.allclose(reduced.vectors.mean(dim=0), torch.zeros(8), atol=1e-5)
    centred = anchors.vectors - anchors.vectors.mean(dim=0)
    basis = torch.linalg.pinv(reduced.vectors) @ centred
    assert torch.allclose(basis @ basis.T, torch.eye(8), atol=1e-4)
    full = anchors.reduce(VOCAB_SIZE - 1)
    assert torch.allclose(full.vectors @ full.vectors.T, centred @ centred.T, atol=1e-3)
    for bad in (-1, VOCAB_SIZE):
        with pytest.raises(ValueError, match="anchor_dim"):
            anchors.reduce(bad)


def test_report(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("INFO"):
        spanning().report(Source.AUDIO)
    assert "audio anchors cumulative explained variance" in caplog.text
    assert "49:" in caplog.text


def test_of(tmp_path: Path) -> None:
    root = DataRoot(tmp_path)
    assert Anchors.of(Source.NONE, root, words(), labels()) is None
    write_targets(root)
    built = Anchors.of(Source.AUDIO, root, words(), labels())
    assert built is not None
    (Targets.directory(root) / "r0.npy").unlink()
    cached = Anchors.of(Source.AUDIO, root, words(), labels())
    assert cached is not None
    assert torch.equal(cached.vectors, built.vectors)
