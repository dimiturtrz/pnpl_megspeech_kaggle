import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.io import wavfile

from megspeech.data.labels.vocabulary import PRIMARY
from megspeech.data.raw.runs import NO_SENTENCE
from megspeech.evaluation.viz.bundle.media import Stimulus
from megspeech.evaluation.viz.bundle.records import SAMPLES, Builder, SampleSet, Window
from megspeech.evaluation.viz.bundle.sample import Sentences

MOCHA = "sub-0_ses-2_task-MOCHATIMIT_run-1"
SHERLOCK = "sub-0_ses-11_task-Sherlock1_run-2"
RATE = 10.0
WINDOW = Window(RATE, 4, offset=1, pad=2)


def words() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "run": [MOCHA, MOCHA, SHERLOCK],
            "sentence": [0, 0, NO_SENTENCE],
            "onset": [0.5, 1.0, 2.0],
            "finish": [0.8, 1.3, 2.4],
            "word": ["the", "cat", "a"],
            "target": [1, 2, 3],
            "at": [5, 10, 20],
        }
    )


def test_samples() -> None:
    assert WINDOW.samples == 8


def test_start() -> None:
    assert WINDOW.start(10) == 9
    assert Window(RATE, 4).start(10) == 10


def sample_set() -> SampleSet:
    keys = words()[["run", "onset"]].assign(id=["0000", "0001", "0002"])
    meg = {"0000": np.ones((2, 3), dtype=np.float32)}
    audio = {"0001": (8, np.array([1, 2], dtype=np.int16))}
    return SampleSet(keys, [{"id": "0000"}], meg, audio)


def test_ids() -> None:
    assert sample_set().ids == ["0000", "0001", "0002"]


def test_write(tmp_path: Path) -> None:
    out = sample_set().write(tmp_path / "set")
    assert json.loads((out / SAMPLES).read_text(encoding="utf-8")) == [{"id": "0000"}]
    assert np.fromfile(out / "meg" / "0000.f32", dtype="<f4").tolist() == [1.0] * 6
    assert wavfile.read(out / "audio" / "0001.wav")[1].tolist() == [1, 2]


def test_identifiers() -> None:
    assert Builder.identifiers(3) == ["0000", "0001", "0002"]
    assert Builder.identifiers(12345)[-1] == "12344"
    assert Builder.identifiers(0) == []


def test_record() -> None:
    builder = Builder(WINDOW, Sentences(words()))
    row = words().iloc[1]
    record = builder.record("0001", row, {"source": "MOCHATIMIT", "length": "short"}, 9, 2)
    assert record["text"] == "the cat"
    assert record["target"] == [4, 7]
    assert record["label"] == PRIMARY.words[2]
    assert (record["subject"], record["session"]) == (0, 2)
    assert record["target_t"] == [1.0, 1.3]
    assert record["meg"] == {"file": "meg/0001.f32", "channels": 2, "times": 8, "sfreq": RATE, "t0": 0.9}
    assert record["audio"] is None
    renamed = words().iloc[2].copy()
    renamed["run"] = "bad"
    with pytest.raises(ValueError, match="bad"):
        builder.record("x", renamed, {"source": "s"}, 0, 1)


def test_build(tmp_path: Path) -> None:
    path = tmp_path / "chapter.wav"
    wavfile.write(path, 10, np.arange(1, 41, dtype=np.int16))
    signals = {MOCHA: np.random.default_rng(0).normal(size=(3, 30)), SHERLOCK: np.ones((3, 30))}
    stimuli = {
        MOCHA: Stimulus(np.zeros(0), np.zeros(0), None),
        SHERLOCK: Stimulus(np.array([2.0]), np.array([2.0]), path),
    }
    built = Builder(WINDOW, Sentences(words())).build(words(), np.array([1, 1, 1, 2, 2, 2, 3, 3, 3]), signals, stimuli)
    assert built.ids == ["0000", "0001", "0002"]
    assert built.keys.columns.tolist() == ["run", "onset", "id"]
    assert built.meg["0000"].shape == (3, 8)
    assert built.meg["0000"].mean(axis=1) == pytest.approx(np.zeros(3), abs=1e-6)
    assert built.meg["0002"].tolist() == np.zeros((3, 8)).tolist()
    assert list(built.audio) == ["0002"]
    assert built.audio["0002"][1].tolist() == list(range(20, 28))
    assert built.records[2]["audio"] == {"file": "audio/0002.wav", "sfreq": 10, "t0": 1.9}
    assert [r["strata"] for r in built.records] == [
        {"source": "MOCHATIMIT", "length": "medium"},
        {"source": "MOCHATIMIT", "length": "medium"},
        {"source": "Sherlock", "length": "short"},
    ]
