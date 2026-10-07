from collections.abc import Mapping
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
from jaxtyping import Float

from megspeech.data.raw.runs import Run
from megspeech.data.raw.serialised import HALF_MAX, SCALE, Convertible, Corpus, Serialised

NAME = "sub-101_ses-0_task-MASC2_run-1"
ATTRS = {"sample_frequency": 250.0}


def signal() -> Float[np.ndarray, "c t"]:
    rng = np.random.default_rng(0)
    return (rng.standard_normal((3, 6000)) * np.array([[1e-12], [5e-13], [2e-11]])).astype(np.float32)


def test_write(tmp_path: Path) -> None:
    meg = signal()
    path = tmp_path / "run_meg.h5"
    Serialised(path).write(meg, ATTRS)
    with h5py.File(path) as stored:
        assert stored["data"].dtype == np.float16
        assert stored["data"].compression == "gzip"
        assert stored["data"].shuffle
        assert stored["data"].chunks == (3, 2500)
        assert stored[SCALE].dtype == np.float32
        assert stored[SCALE][:].tolist() == pytest.approx(meg.std(axis=1).tolist(), rel=1e-5)
        assert stored.attrs["sample_frequency"] == 250.0


def test_write_flat_and_saturated(tmp_path: Path) -> None:
    meg = signal()
    meg[0] = 0.0
    meg[1, 0] = 1e6 * meg[1].std()
    path = tmp_path / "run_meg.h5"
    Serialised(path).write(meg, ATTRS)
    with h5py.File(path) as stored:
        assert stored[SCALE][0] == 1.0
        assert np.isfinite(stored["data"][:]).all()
        assert stored["data"][1, 0] <= HALF_MAX


def test_read(tmp_path: Path) -> None:
    meg = signal()
    path = tmp_path / "run_meg.h5"
    Serialised(path).write(meg, ATTRS)
    with h5py.File(path) as stored:
        read = Serialised.read(stored)
    assert read.dtype == np.float32
    assert np.abs(read - meg).max(axis=1) / meg.std(axis=1) == pytest.approx(0, abs=0.02)


def test_read_legacy(tmp_path: Path) -> None:
    meg = signal()
    path = tmp_path / "legacy_meg.h5"
    with h5py.File(path, "w") as stored:
        stored["data"] = meg
    with h5py.File(path) as stored:
        assert np.array_equal(Serialised.read(stored), meg)


class Fake:
    name = NAME

    def written(self, target: float) -> tuple[Float[np.ndarray, "c t"], Mapping[str, object]]:
        return signal(), {"target": target}

    def events(self) -> pd.DataFrame:
        return pd.DataFrame({"kind": ["word"]})


def corpus(tmp_path: Path) -> Corpus[float]:
    return Corpus(tmp_path / "in", tmp_path / "out")


def test_meg(tmp_path: Path) -> None:
    assert corpus(tmp_path).meg(Fake()) == tmp_path / "out" / "serialised" / f"{NAME}_meg.h5"


def test_events(tmp_path: Path) -> None:
    assert corpus(tmp_path).events(Fake()) == tmp_path / "out" / "events" / f"{NAME}_events.tsv"


def test_convert(tmp_path: Path) -> None:
    recording: Convertible[float] = Fake()
    study = corpus(tmp_path)
    meg = study.convert(recording, 1.5)
    with h5py.File(meg) as stored:
        assert stored["data"].shape == (3, 6000)
        assert stored.attrs["target"] == 1.5
    [run] = Run.discover(study.target)
    assert run.name == NAME
    meg.unlink()
    assert study.convert(recording, 1.5) == meg
    assert not meg.exists()
