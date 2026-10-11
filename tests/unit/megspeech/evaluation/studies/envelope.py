from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

from megspeech.data.constants import SFREQ
from megspeech.data.labels.vocabulary import IGNORE
from megspeech.data.raw.runs import Run
from megspeech.data.spans.grouping import Grouping
from megspeech.evaluation.studies.envelope import (
    BINS,
    WINDOW,
    Band,
    Cohort,
    Covariance,
    Envelope,
    Extractor,
    Mechanism,
    Recording,
    Source,
    Study,
)
from megspeech.evaluation.studies.shrunklda import Harness
from tests.unit.megspeech.evaluation.studies.shrunklda import classes, separable

CHANNELS = 6
SAMPLES = 2000


def signal() -> np.ndarray:
    return np.random.default_rng(0).standard_normal((CHANNELS, SAMPLES)).astype(np.float32)


def test_parse() -> None:
    """A spec reads as its edges."""
    assert Band.parse("4-8") == Band(4.0, 8.0)


def test_label() -> None:
    assert Band.parse("30-45").label == "30-45Hz"


def test_filtered() -> None:
    """A tone inside the band survives, one outside is cut."""
    t = np.arange(SAMPLES) / SFREQ
    inside, outside = np.sin(2 * np.pi * 40 * t), np.sin(2 * np.pi * 5 * t)
    kept = Band(30, 45).filtered(np.stack([inside, outside]))
    assert kept[0].std() > 0.5
    assert kept[1].std() < 0.05


def test_covariance_of() -> None:
    cov = Covariance.of(signal())
    assert cov.total.shape == (CHANNELS, CHANNELS)
    assert cov.seen == SAMPLES


def test_merge() -> None:
    cov = Covariance.of(signal())
    merged = cov.merge(cov)
    assert merged.seen == 2 * SAMPLES
    assert np.allclose(merged.total, 2 * cov.total)


def test_whitener() -> None:
    """Projected components have unit variance and no correlation."""
    x = signal()
    comps = Covariance.of(x).whitener(3).T @ (x - x.mean(axis=1, keepdims=True))
    assert np.allclose(np.cov(comps), np.eye(3), atol=1e-3)


def test_log() -> None:
    meg = signal()
    envelope = Envelope(Band(30, 45), Covariance.of(meg).whitener(3))
    out = envelope.log(meg)
    assert out.shape == (3, SAMPLES)
    assert np.allclose(out.mean(axis=1), 0, atol=1e-4)


def test_binned() -> None:
    meg = signal()
    envelope = Envelope(Band(30, 45), Covariance.of(meg).whitener(3))
    onsets = np.array([10, 500, 900])
    out = envelope.binned(meg, onsets)
    assert out.shape == (3, 3, BINS)
    assert np.allclose(out[0, 0, 0], envelope.log(meg)[0, 10 : 10 + WINDOW // BINS].mean(), atol=1e-4)


def test_novel() -> None:
    """Only labelled words of a multi-word span with unheard text are kept."""
    words = pd.DataFrame(
        {
            "run": "a",
            "onset": [0.0, 0.3, 0.6, 5.0, 5.3],
            "word": ["x", "y", "z", "p", "q"],
            "sentence": [0, 0, 0, 1, 1],
            "target": [1, IGNORE, 2, 3, 4],
            "token": [10, 11, 12, 13, 14],
        }
    )
    kept = Cohort.novel(words, Grouping.SENTENCE, frozenset({(13, 14)}))
    assert list(kept["token"]) == [10, 12]


def test_cohort_of() -> None:
    """A domain below `MIN_NOVEL` words does not count."""
    words = pd.DataFrame(
        {"run": "a", "onset": [0.0, 0.3], "word": ["x", "y"], "sentence": [0, 0], "target": [1, 2], "token": [1, 2]}
    )
    cohort = Cohort.of(words, {"small": words}, Grouping.SENTENCE, frozenset())
    assert len(cohort.train) == 2
    assert cohort.domains == {}


def test_table() -> None:
    words = pd.DataFrame({"run": "a", "onset": [0.0], "target": [1]})
    cohort = Cohort(words, {"d": words})
    assert list(cohort.table()["part"]) == ["train", "d"]


def test_mechanism_of() -> None:
    """Pure noise sits at the chance ratio; a class-locked feature far above it."""
    n = 4000
    y = classes(n)
    noise = torch.randn(n, 3, 2, generator=torch.Generator().manual_seed(1))
    flat = Mechanism.of(noise, y)
    assert flat.ratio.max() < 5 * flat.chance
    locked = noise.clone()
    locked[:, :, 0] += y.float()[:, None]
    assert Mechanism.of(locked, y).ratio[0] > 100 * flat.chance


def test_summary() -> None:
    noise = torch.randn(400, 3, 2, generator=torch.Generator().manual_seed(2))
    assert "vs chance" in Mechanism.of(noise, classes(400)).summary()


def stored(directory: Path, index: int) -> Run:
    path = directory / f"sub-0_ses-{index}_task-T_run-1_meg.h5"
    with h5py.File(path, "w") as out:
        out["data"] = np.random.default_rng(index).standard_normal((CHANNELS, SAMPLES)).astype(np.float32)
        out.attrs["sample_frequency"] = float(SFREQ)
    return Run(path)


def test_load(tmp_path: Path) -> None:
    run = stored(tmp_path, 1)
    assert Recording.load(run).meg.shape == (CHANNELS, SAMPLES)


def test_covariances(tmp_path: Path) -> None:
    out = Source(stored(tmp_path, 1)).covariances((Band(30, 45), Band(4, 8)))
    assert len(out) == 2
    assert out[0].seen == SAMPLES


def test_features(tmp_path: Path) -> None:
    run = stored(tmp_path, 1)
    envelope = Envelope(Band(30, 45), Covariance.of(signal()).whitener(3))
    out = Source(run).features(np.array([10, 20]), (envelope,))
    assert out.shape == (2, 1, 3, BINS)


def test_projections(tmp_path: Path) -> None:
    runs = {run.name: run for run in (stored(tmp_path, 1), stored(tmp_path, 2))}
    out = Extractor((Band(30, 45),), runs, 1).projections(list(runs), 3)
    assert out[0].matrix.shape == (CHANNELS, 3)


def test_extract(tmp_path: Path) -> None:
    runs = {run.name: run for run in (stored(tmp_path, 1), stored(tmp_path, 2))}
    names = list(runs)
    extractor = Extractor((Band(30, 45),), runs, 1)
    envelopes = extractor.projections(names, 3)
    table = pd.DataFrame({"run": [names[1], names[0], names[1]], "onset": [0.1, 0.2, 0.3]})
    out = extractor.extract(table, envelopes)
    assert out.shape[:2] == (3, 1)
    assert out.shape[3] == BINS


def test_datasets() -> None:
    n = 12
    table = pd.DataFrame({"run": ["a"] * n, "onset": np.arange(n) * 1.0, "target": classes(n).numpy()})
    cohort = Cohort(table.iloc[:8], {"d": table.iloc[8:]})
    features = np.zeros((n, 2, 3, BINS), dtype=np.float32)
    train, domains = Study.datasets(cohort, features, [1])
    assert train.x.shape == (8, 3 * BINS)
    assert domains["d"].x.shape == (4, 3 * BINS)


def test_report() -> None:
    data = separable()
    line = Study.report("band", Harness(), data, {"d": data})
    assert line.startswith("band: macro")


def test_main() -> None:
    """The CLI entry exists and parses; the full run is exercised on real data only."""
    assert callable(Study.main)
