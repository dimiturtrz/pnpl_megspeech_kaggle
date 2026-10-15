from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from megspeech.evaluation.viz.bundle.media import PCM, Meg, Stimulus
from megspeech.objective.targets.speech import LAG, SAMPLING
from tests.unit.megspeech.data.config import HELD
from tests.unit.megspeech.objective.targets.speech import heard_run

RATE = 10.0


def signal() -> np.ndarray:
    return np.arange(12, dtype=np.float32).reshape(2, 6)


def test_window() -> None:
    meg = Meg(signal(), RATE)
    assert meg.window(1, 3).tolist() == [[1, 2, 3], [7, 8, 9]]
    assert meg.window(-2, 4).tolist() == [[0, 0, 0, 1], [0, 0, 6, 7]]
    assert meg.window(5, 3).tolist() == [[5, 0, 0], [11, 0, 0]]
    assert meg.window(9, 2).tolist() == [[0, 0], [0, 0]]


def test_zscored() -> None:
    window = np.array([[1.0, 2.0, 3.0], [4.0, 4.0, 4.0]])
    scored = Meg.zscored(window)
    assert scored.dtype == np.float32
    assert scored[0].mean() == pytest.approx(0)
    assert scored[0].std() == pytest.approx(1)
    assert scored[1].tolist() == [0, 0, 0]


def test_meg_write(tmp_path: Path) -> None:
    path = Meg.write(tmp_path / "meg" / "0000.f32", signal())
    assert np.fromfile(path, dtype="<f4").reshape(2, 6).tolist() == signal().tolist()


def test_of(tmp_path: Path) -> None:
    heard = Stimulus.of(heard_run(tmp_path))
    assert heard.onsets.tolist() == pytest.approx([10.94, 13.8])
    assert heard.chapter.tolist() == pytest.approx([0.94 + LAG, 3.8 + LAG])
    assert heard.path is not None
    assert heard.path.name == "sherlock1_01_doyle_64kb.wav"
    assert Stimulus.of(heard_run(tmp_path, HELD, audio=False)).path is None


def test_sound(tmp_path: Path) -> None:
    rate, sound = Stimulus.of(heard_run(tmp_path)).sound
    assert rate == SAMPLING
    assert sound.dtype == np.int16
    assert Stimulus(np.zeros(0), np.zeros(0), None).sound[1].size == 0


def test_pcm() -> None:
    stereo = np.array([[0.5, 0.5], [-1.0, -1.0], [2.0, 2.0]])
    assert Stimulus.pcm(stereo).tolist() == [round(0.5 * PCM), -PCM, PCM]
    assert Stimulus.pcm(np.array([3, -3], dtype=np.int16)).tolist() == [3, -3]


def test_at() -> None:
    stimulus = Stimulus(np.array([10.0, 20.0]), np.array([1.0, 5.0]), None)
    assert stimulus.at(11.0) == pytest.approx(2.0)
    assert stimulus.at(19.5) == pytest.approx(4.5)


def test_clip(tmp_path: Path) -> None:
    path = tmp_path / "chapter.wav"
    wavfile.write(path, 10, np.arange(1, 31, dtype=np.int16))
    stimulus = Stimulus(np.array([100.0]), np.array([1.0]), path)
    assert stimulus.clip(100.0, 0.5) == (10, pytest.approx(np.arange(11, 16)))
    clipped = stimulus.clip(98.5, 1.0)
    assert clipped is not None
    assert clipped[1].tolist() == [0] * 5 + [1, 2, 3, 4, 5]
    assert Stimulus(np.zeros(0), np.zeros(0), None).clip(0.0, 1.0) is None


def test_stimulus_write(tmp_path: Path) -> None:
    path = Stimulus.write(tmp_path / "audio" / "0000.wav", 8, np.array([1, -2, 3]))
    rate, read = wavfile.read(path)
    assert rate == 8
    assert read.tolist() == [1, -2, 3]
