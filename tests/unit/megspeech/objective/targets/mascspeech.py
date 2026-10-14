from pathlib import Path
from typing import cast

import numpy as np
import pytest
from scipy.io import wavfile

from core.paths import DataRoot
from megspeech.data.config import Cut
from megspeech.data.raw.fetch import MASC
from megspeech.data.raw.runs import Run
from megspeech.data.raw.sources.masc import Masc, Recording
from megspeech.objective.targets import mascspeech as module
from megspeech.objective.targets import speech
from megspeech.objective.targets.loudness import CUT, LENGTH, Loudness
from megspeech.objective.targets.mascspeech import Told, Wavs
from megspeech.objective.targets.speech import SAMPLING, STEM, Targets, Teacher
from tests.unit.megspeech.data.raw.runs import meg
from tests.unit.megspeech.data.raw.sources.masc import NAME, SOUNDS, recording
from tests.unit.megspeech.objective.targets.scansloudness import root
from tests.unit.megspeech.objective.targets.speech import CPU, DIM, SECONDS, Loader, Root, loud


def told(data: DataRoot, *, dropped: bool = False) -> Told:
    """A downloaded recording whose two words play in two wavs, silent until 1 s, and the run converted from it; its
    events lack the last word if `dropped`."""
    heard = recording(Masc.under(data).source)
    for sound in SOUNDS:
        wav = heard.home / sound
        wav.parent.mkdir(parents=True, exist_ok=True)
        wavfile.write(wav, SAMPLING, (loud(SECONDS, 1.0).audio * 1000).astype(np.int16))
    run = Run(meg(data.processed(MASC.dataset), NAME))
    events = heard.events()
    events = events.iloc[:-1] if dropped else events
    run.events.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(run.events, sep="\t", index=False)
    return Told(heard, run)


def test_targets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = root(tmp_path, monkeypatch)
    targets = told(data).targets(Wavs.envelope(CUT))
    assert targets.vectors.shape == (2, LENGTH)
    assert targets.present[0]
    with pytest.raises(ValueError, match="heard words"):
        told(root(tmp_path / "dropped", monkeypatch), dropped=True).targets(Wavs.envelope(CUT))


def test_chapter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    heard = told(root(tmp_path, monkeypatch)).recording
    wavs = Wavs.envelope(CUT)
    chapter = wavs.chapter(heard.home / SOUNDS[0])
    assert len(chapter.audio) == SECONDS * SAMPLING
    assert wavs.chapter(heard.home / SOUNDS[0]) is chapter


def test_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    heard = told(root(tmp_path, monkeypatch)).recording
    made: list[Recording] = []
    wavs = Wavs(lambda _: made.append(heard) or (lambda onset: Targets(np.zeros((len(onset), 1), np.float16))))
    cutter = wavs(heard.home / SOUNDS[0])
    assert wavs(heard.home / SOUNDS[0]) is cutter
    assert len(made) == 1
    assert cutter(np.array([1.0, 2.0])).vectors.shape == (2, 1)


def test_envelope(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    heard = told(root(tmp_path, monkeypatch)).recording
    assert Wavs.envelope(CUT)(heard.home / SOUNDS[0])(np.array([1.0, 2.0])).vectors.shape == (2, LENGTH)


def test_speech(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    heard = told(root(tmp_path, monkeypatch)).recording
    monkeypatch.setitem(speech.LOADERS, Teacher.WAV2VEC2, Loader())
    wavs = Wavs.speech(cast("DataRoot", Root(tmp_path)), CPU)
    assert wavs(heard.home / SOUNDS[0])(np.array([1.0, 2.0])).vectors.shape == (2, DIM)


def test_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = root(tmp_path, monkeypatch)
    told(data)
    directory = Loudness.directory(data, CUT)
    assert Wavs.write(data, directory, Wavs.envelope(CUT)) == directory
    assert [path.stem for path in directory.glob("*.npy")] == [NAME]
    written = (directory / f"{NAME}.npy").stat().st_mtime_ns
    Wavs.write(data, directory, Wavs.envelope(CUT))
    assert (directory / f"{NAME}.npy").stat().st_mtime_ns == written


def test_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = root(tmp_path, monkeypatch)
    told(data)
    monkeypatch.setitem(speech.LOADERS, Teacher.WAV2VEC2, Loader())
    monkeypatch.setattr(module, "DEVICE", "cpu")
    monkeypatch.setattr("sys.argv", ["mascspeech", "--window", "1.5"])
    Wavs.main()
    assert np.load(Loudness.directory(data, Cut(75, 0)) / f"{NAME}.npy").shape == (2, 75)
    assert not (Targets.directory(data) / f"{NAME}.npy").exists()
    monkeypatch.setattr("sys.argv", ["mascspeech", "--window", "1.5", "--speech"])
    Wavs.main()
    assert np.load(Targets.directory(data) / f"{NAME}.npy").shape == (2, DIM)
    assert Targets.directory(data).name == STEM
