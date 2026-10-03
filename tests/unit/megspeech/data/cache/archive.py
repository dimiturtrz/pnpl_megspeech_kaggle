from pathlib import Path

import numpy as np
import pytest

from megspeech.data.cache.archive import Archive, ArchiveKind, Epochs
from megspeech.data.cache.run_cache import RunCache
from megspeech.data.constants import RATE, SFREQ, WINDOW

CHANNELS = 3
SLOTS = 9
SAMPLES = 4 * SFREQ
LEAD = 300
ONSETS = [0.5, 1.02, 2.0, 2.98]
"""Seconds into the stretch; none lands on a half sample at `RATE`, where a sum of float offsets could round apart."""

WORD_EPOCHS = 2


def stretch(seed: int = 0, samples: int = SAMPLES) -> np.ndarray:
    return (np.random.default_rng(seed).normal(size=(CHANNELS, samples)) * 1e-12).astype(np.float32)


def sentence_archive(raw: Path, subject: int = 0, rate: float = SFREQ, onsets: list[float] = ONSETS) -> Path:
    """A lead-in sentence of `LEAD` samples holding one word, then the stretch holding `onsets`, zero-padded."""
    meg = np.zeros((2, CHANNELS, SAMPLES), dtype=np.float32)
    meg[0, :, :LEAD] = stretch(1, LEAD)
    meg[1] = stretch()
    onset = np.full((2, SLOTS), np.nan, dtype=np.float32)
    onset[0, 0] = 0.1
    onset[1, : len(onsets)] = onsets
    path = raw / f"subj{subject:02d}_{ArchiveKind.SENTENCE}.npz"
    np.savez(
        path,
        meg=meg,
        sentence_n_times=np.array([LEAD, SAMPLES]),
        word_onsets_s=onset,
        word_mask=~np.isnan(onset),
        sfreq=np.float64(rate),
    )
    return path


def word_archive(raw: Path, subject: int = 0) -> Path:
    path = raw / f"subj{subject:02d}_{ArchiveKind.WORD}.npz"
    meg = np.stack([stretch(2 + epoch, WINDOW) for epoch in range(WORD_EPOCHS)])
    np.savez(path, meg=meg, sfreq=np.float64(SFREQ), tmin=np.float64(0), tmax=np.float64(1))
    return path


def test_sentences(tmp_path: Path) -> None:
    with np.load(sentence_archive(tmp_path)) as stored:
        epochs = Epochs.sentences(stored)
    assert epochs.length.tolist() == [LEAD, SAMPLES]
    assert epochs.mask.sum(axis=1).tolist() == [1, len(ONSETS)]
    assert epochs.onset[1, : len(ONSETS)] == pytest.approx(ONSETS)


def test_epochs_words(tmp_path: Path) -> None:
    with np.load(word_archive(tmp_path)) as stored:
        epochs = Epochs.words(stored)
    assert epochs.length.tolist() == [WINDOW] * WORD_EPOCHS
    assert epochs.onset.tolist() == [[0.0]] * WORD_EPOCHS
    assert epochs.mask.all()


def test_archive_words(tmp_path: Path) -> None:
    table = Archive(sentence_archive(tmp_path)).words(RunCache(tmp_path))
    assert table["sentence"].tolist() == [0] + [1] * len(ONSETS)
    assert table["onset"].iloc[1:].to_numpy() == pytest.approx(LEAD / SFREQ + np.array(ONSETS))
    assert set(table["word"]) == {""}
    assert table["end"].to_numpy() == pytest.approx((LEAD + np.r_[0, [SAMPLES] * len(ONSETS)]) / SFREQ)
    ties = np.array([0.31, 0.33, 0.35], dtype=np.float32)
    tied = Archive(sentence_archive(tmp_path, onsets=ties.tolist())).words(RunCache(tmp_path))
    lead = RunCache(Path()).length(np.array([LEAD]))[0]
    sample = np.round(tied["onset"].to_numpy()[1:] * RATE) - lead
    assert sample.tolist() == np.round(ties.astype(np.float64) * RATE).tolist()
    with pytest.raises(ValueError, match="runs past"):
        Archive(sentence_archive(tmp_path, onsets=[3.5])).words(RunCache(tmp_path))


def test_name(tmp_path: Path) -> None:
    assert Archive(word_archive(tmp_path)).name == f"subj00_{ArchiveKind.WORD}"


def test_epochs(tmp_path: Path) -> None:
    assert Archive(sentence_archive(tmp_path)).epochs().length.tolist() == [LEAD, SAMPLES]
    with pytest.raises(ValueError, match="1000"):
        Archive(sentence_archive(tmp_path, rate=1000)).epochs()


def test_signal(tmp_path: Path) -> None:
    """Each epoch is decimated over its own length only — the zero padding after it never reaches the cache."""
    cache = RunCache(tmp_path)
    signal = Archive(sentence_archive(tmp_path)).signal(cache)
    lead = cache.length(np.array([LEAD]))[0]
    assert signal.shape == (CHANNELS, lead + SAMPLES // 5)
    assert np.array_equal(signal[:, :lead], cache.decimated(stretch(1, LEAD)))
    assert np.array_equal(signal[:, lead:], cache.decimated(stretch()))
