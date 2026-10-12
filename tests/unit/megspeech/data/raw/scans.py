import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mne
import numpy as np
import pytest
from scipy.io import wavfile

from core.paths import ENV_OVERRIDE, DataRoot
from megspeech.data.labels.vocabulary import ABSENT, PRIMARY, SECONDARY
from megspeech.data.raw import scans as module
from megspeech.data.raw.events import Event
from megspeech.data.raw.kymatasync import Alignment, Clock, Envelope
from megspeech.data.raw.masc import COLUMNS, Neuromag
from megspeech.data.raw.scans import GERMAN, SCANS, Book, Recording, Scans, main

FIF = "sub-03_task-audiobook2_run-01_meg.fif"
NAME = "sub-403_ses-1_task-SCANSab2_run-1"
GRID = """File type = "ooTextFile"
Object class = "TextGrid"

item []:
    item [1]:
        class = "IntervalTier"
        name = "words"
        intervals: size = 5
        intervals [1]:
            xmin = 0.0
            xmax = 0.5
            text = ""
        intervals [2]:
            xmin = 0.5
            xmax = 0.8
            text = "so"
        intervals [3]:
            xmin = 0.8
            xmax = 1.2
            text = "in"
        intervals [4]:
            xmin = 1.2
            xmax = 2.5
            text = ""
        intervals [5]:
            xmin = 2.5
            xmax = 3.0
            text = "wahr"
    item [2]:
        class = "IntervalTier"
        name = "phones"
        intervals: size = 1
        intervals [1]:
            xmin = 0.0
            xmax = 3.0
            text = ""
"""
RATE = 8000
OFFSET = 0.02
DRIFT = 0.0005


def dataset(tmp_path: Path) -> Recording:
    """A downloaded recording with its sidecars and its part's stimulus."""
    fif = tmp_path / "sub-03" / "meg" / FIF
    fif.parent.mkdir(parents=True)
    fif.write_bytes(b"fif")
    fif.with_name("sub-03_task-audiobook2_run-01_events.tsv").write_text("onset\n1.0\n")
    stimuli = tmp_path / "stimuli" / "audiobooks"
    stimuli.mkdir(parents=True)
    (stimuli / "task-audiobook2_run-01_stim.TextGrid").write_text(GRID, encoding="utf-8")
    wavfile.write(
        stimuli / "task-audiobook2_run-01_stim.wav",
        RATE,
        np.random.default_rng(0).standard_normal(3 * RATE).astype(np.float32),
    )
    return Recording(fif)


def test_words(tmp_path: Path) -> None:
    words = dataset(tmp_path).book.words
    assert words["segment"].tolist() == ["so", "in", "wahr"]
    assert words["timemeg"].tolist() == [0.5, 0.8, 2.5]


def test_alignment(tmp_path: Path) -> None:
    """A mono wav, as the dataset ships, reads as one channel."""
    assert len(dataset(tmp_path).book.alignment.reference.samples) == 3 * 1000


def test_parsed(tmp_path: Path) -> None:
    assert Recording(tmp_path / FIF).parsed == {"subject": 403, "book": 2, "run": 1}
    with pytest.raises(ValueError, match="sub-X1"):
        _ = Recording(tmp_path / "sub-X1_task-audiobook1_run-01_meg.fif").parsed


def test_name(tmp_path: Path) -> None:
    assert Recording(tmp_path / FIF).name == NAME


def test_shared(tmp_path: Path) -> None:
    shared = Recording(tmp_path / "sub-03" / "meg" / FIF).shared
    assert shared["cross_talk"] == tmp_path / "sub-03" / "meg" / "sub-03_acq-crosstalk_meg.fif"
    assert shared["calibration"] == tmp_path / "sub-03" / "meg" / "sub-03_acq-calibration_meg.dat"


def test_audio_channel(tmp_path: Path) -> None:
    assert Recording(tmp_path / FIF).audio_channel == module.AUDIO_CHANNEL


def test_prefix(tmp_path: Path) -> None:
    assert Recording(tmp_path / FIF).prefix == GERMAN


def test_book(tmp_path: Path) -> None:
    rec = dataset(tmp_path)
    assert rec.book == Book(tmp_path / "stimuli" / "audiobooks" / "task-audiobook2_run-01_stim")


def test_plays(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rec = dataset(tmp_path)
    monkeypatch.setattr(Recording, "audio", lambda _self: Envelope(np.zeros(10)))
    monkeypatch.setattr(Alignment, "locate", lambda _self, _recorded, plays: [3.0] * plays)
    monkeypatch.setattr(Alignment, "clock", lambda _self, _recorded, nominal, _offsets: Clock(nominal, OFFSET, DRIFT))
    assert rec.plays() == [Clock(3.0, OFFSET, DRIFT)]
    monkeypatch.setattr(
        Alignment, "clock", lambda _self, _recorded, nominal, _offsets: Clock(nominal, 0, 0, fitted=False)
    )
    with pytest.raises(ValueError, match="not believable"):
        rec.plays()


def test_events_at(tmp_path: Path) -> None:
    events = dataset(tmp_path).events_at([Clock(10.0, OFFSET, DRIFT)])
    assert list(events.columns) == [*COLUMNS, Event.CHAPTER]
    assert set(events["kind"]) == {"word"}
    assert events["segment"].tolist() == [GERMAN + word for word in ("so", "in", "wahr")]
    assert events["timechapter"].tolist() == [0.5, 0.8, 2.5]
    assert events["timemeg"].iloc[0] == pytest.approx(10.0 + OFFSET + (1 + DRIFT) * 0.5)
    assert events["sentenceidx"].tolist() == [0, 0, 1]


def test_events_out_of_vocabulary(tmp_path: Path) -> None:
    """`so` and `in` are English vocabulary words; none of the German words spelled so is."""
    segments = dataset(tmp_path).events_at([Clock(0.0, 0.0, 0.0)])["segment"]
    assert (PRIMARY.labels(segments) == ABSENT).all()
    assert (SECONDARY.labels(segments) == ABSENT).all()
    assert (PRIMARY.labels(["so", "in"]) != ABSENT).all()


def test_under(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_OVERRIDE, str(tmp_path))
    root = DataRoot.from_config()
    assert Scans.under(root) == Scans(SCANS.home(root), root.processed(SCANS.dataset))


def test_recordings(tmp_path: Path) -> None:
    ready = dataset(tmp_path)
    unfinished = tmp_path / "sub-04" / "meg" / "sub-04_task-audiobook1_run-01_meg.fif"
    unfinished.parent.mkdir(parents=True)
    unfinished.touch()
    olsa = tmp_path / "sub-03" / "meg" / "sub-03_task-olsa_meg.fif"
    olsa.write_bytes(b"fif")
    study = Scans(tmp_path, tmp_path / "out")
    assert list(study.recordings()) == [ready]
    assert list(study.recordings((403,))) == [ready]
    assert list(study.recordings((404,))) == []


def test_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_OVERRIDE, str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["scans", "--workers", "1", "--target", str(tmp_path / "scratch")])
    rec = dataset(Scans.under(DataRoot.from_config()).source)
    converted: list[Recording] = []
    target = Neuromag(mne.create_info(["MEG0111"], 1000.0, "mag"))
    monkeypatch.setattr(module, "ProcessPoolExecutor", ThreadPoolExecutor)
    monkeypatch.setattr(Neuromag, "of", classmethod(lambda cls, path: target))
    monkeypatch.setattr(Scans, "convert", lambda self, recording, _target: converted.append(recording))
    main()
    assert converted == [rec]
