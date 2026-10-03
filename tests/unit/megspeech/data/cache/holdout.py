"""The holdout's windows against the training path's: one 250 Hz stretch, cached as a training run and as a holdout
sentence, gathers bit-identical windows at the same words."""

import logging
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import torch
from pnpl.competition.holdout import HOLDOUT_SUBDIR, LibriBrainCompetitionHoldout

from core.paths import ENV_OVERRIDE, DataRoot
from megspeech.data.cache.holdout import HOLDOUT_DIR, Archive, ArchiveKind, Epochs, HoldoutCache, Track, main
from megspeech.data.cache.run_cache import DIRECTORY, TAIL, WORD_TABLE, DeviceRecording, RunCache
from megspeech.data.constants import RATE, SFREQ, WINDOW
from megspeech.data.raw.runs import Run

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


def holdout(raw: Path) -> HoldoutCache:
    """Both archives of subject 0 where the competition lays them out under the `raw` root."""
    archives = raw / HOLDOUT_SUBDIR
    archives.mkdir(parents=True, exist_ok=True)
    sentence_archive(archives)
    word_archive(archives)
    return HoldoutCache(archives, RunCache(raw.parent / HOLDOUT_DIR))


def training(root: Path) -> Run:
    """The stretch as a training run whose events put a word at each of `ONSETS`, the last ending `TAIL` before the
    stretch does, as the holdout's sentence epoch ends `TAIL` after its sentence's last word."""
    run = Run(root / "derivatives" / "serialised" / "sub-0_ses-1_task-Sherlock1_run-1_meg.h5")
    run.meg.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(run.meg, "w") as stored:
        stored["data"] = stretch()
        stored.attrs["sample_frequency"] = float(SFREQ)
    run.events.parent.mkdir(parents=True, exist_ok=True)
    duration = SAMPLES / SFREQ - TAIL - ONSETS[-1]
    rows = {"kind": "word", "segment": "word", "sentenceidx": 0, "timemeg": ONSETS, "duration": duration}
    pd.DataFrame(rows).to_csv(run.events, sep="\t", index=False)
    return run


def test_train_and_holdout_windows_match(tmp_path: Path) -> None:
    trained = RunCache(tmp_path / "train")
    trained.ready()
    run = training(tmp_path)
    trained.cache_run(run)
    trained.publish(RunCache.words(run))
    source = DeviceRecording.load(trained, "cpu")
    cache = holdout(tmp_path / "raw")
    cache.build()
    every = torch.arange(len(ONSETS))
    stretch_rows = 1 + every
    for length in (RATE, 2 * RATE):
        held = cache.windows("cpu", length)(stretch_rows)
        assert torch.equal(held, source.windows(length)(every))
        assert held.shape == (len(ONSETS), CHANNELS, length)
    past = SAMPLES // (SFREQ // RATE) - round(ONSETS[-1] * RATE)
    tail = cache.windows("cpu", 2 * RATE)(stretch_rows[-1:])
    assert tail[..., past:].eq(0).all()
    assert tail[..., :past].ne(0).any(dim=-1).all()


def test_rows_follow_competition_order(tmp_path: Path) -> None:
    cache = holdout(tmp_path / "raw")
    table = cache.build()
    frame = LibriBrainCompetitionHoldout(subjects=[0], data_path=tmp_path / "raw", download=False).metadata_frame()
    assert len(table) == len(frame) == len(cache.windows("cpu", RATE))
    lead = RunCache(Path()).length(np.array([LEAD]))[0] / RATE
    start = np.where(frame["source"] == "sentence", frame["epoch"] * lead, 0.0)
    assert table["sentence"].tolist() == frame["epoch"].tolist()
    assert np.allclose(table["onset"].to_numpy()[: 1 + len(ONSETS)], (start + frame["onset_s"])[: 1 + len(ONSETS)])


def test_of(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_OVERRIDE, str(tmp_path))
    cache = HoldoutCache.of(DataRoot.from_config())
    assert cache.raw.name == HOLDOUT_SUBDIR
    assert cache.cache.cache.parts[-2:] == (DIRECTORY.format(rate=RATE), HOLDOUT_DIR)
    fast = HoldoutCache.of(DataRoot.from_config(), 2).cache
    assert (fast.cache.parts[-2:], fast.rate) == (("continuous125", HOLDOUT_DIR), SFREQ / 2)
    assert HoldoutCache.of(DataRoot.from_config(), track=Track.BROAD).subjects == tuple(range(1, 40))


def test_subjects() -> None:
    assert Track.DEEP.subjects == (0,)
    assert Track.BROAD.subjects == tuple(range(1, 40))


def test_holdout(tmp_path: Path) -> None:
    holdout(tmp_path)
    assert len(Track.DEEP.holdout(tmp_path)) == 1 + len(ONSETS) + WORD_EPOCHS


def test_archives(tmp_path: Path) -> None:
    names = [archive.name for archive in HoldoutCache(tmp_path, RunCache(tmp_path), (1, 0)).archives()]
    assert names == [
        f"subj01_{ArchiveKind.SENTENCE}",
        f"subj01_{ArchiveKind.WORD}",
        f"subj00_{ArchiveKind.SENTENCE}",
        f"subj00_{ArchiveKind.WORD}",
    ]


def test_build(tmp_path: Path) -> None:
    cache = holdout(tmp_path / "raw")
    table = cache.build()
    assert len(table) == 1 + len(ONSETS) + WORD_EPOCHS
    assert (cache.cache.cache / WORD_TABLE).exists()
    assert not list(cache.cache.cache.glob("*.partial"))
    assert table["run"].tolist()[-WORD_EPOCHS:] == [f"subj00_{ArchiveKind.WORD}"] * WORD_EPOCHS
    assert np.load(cache.cache.path(f"subj00_{ArchiveKind.WORD}")).shape == (CHANNELS, WORD_EPOCHS * RATE)


def test_windows(tmp_path: Path) -> None:
    cache = holdout(tmp_path / "raw")
    cache.build()
    windows = cache.windows("cpu", RATE)
    assert len(windows) == 1 + len(ONSETS) + WORD_EPOCHS
    assert torch.isfinite(windows(torch.arange(len(windows)))).all()
    lead = RunCache(Path()).length(np.array([LEAD]))[0] - round(0.1 * RATE)
    long = cache.windows("cpu", 2 * RATE)(torch.tensor([0, len(windows) - 1]))
    assert long[0, :, lead:].eq(0).all()
    assert long[1, :, RATE:].eq(0).all()


def test_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    monkeypatch.setenv(ENV_OVERRIDE, str(tmp_path))
    root = DataRoot.from_config()
    holdout(root.raw())
    monkeypatch.setattr(sys, "argv", ["holdout"])
    with caplog.at_level(logging.INFO):
        main()
    rows = 1 + len(ONSETS) + WORD_EPOCHS
    assert (
        f"{rows} windows of ({CHANNELS}, {RATE}) for {rows} holdout rows, finite 1.000000, zeroed 0.000000"
        in caplog.text
    )
    monkeypatch.setattr(sys, "argv", ["holdout", "--window", "2"])
    with caplog.at_level(logging.INFO):
        main()
    assert f"{rows} windows of ({CHANNELS}, {2 * RATE}) for {rows} holdout rows" in caplog.text


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
