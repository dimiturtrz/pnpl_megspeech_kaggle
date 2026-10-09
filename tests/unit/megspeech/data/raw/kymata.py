import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import pytest
from scipy.io import wavfile

from core.paths import ENV_OVERRIDE, DataRoot
from megspeech.data.constants import SFREQ
from megspeech.data.raw import kymata as module
from megspeech.data.raw.kymata import KYMATA, Clip, Kymata, Recording, Study, main
from megspeech.data.raw.kymatasync import Alignment, Clock, Envelope
from megspeech.data.raw.masc import COLUMNS, HIGHPASS, Neuromag

FIF = "sub-E3b_task-en_run-2_meg.fif"
NAME = "sub-323_ses-1_task-Kymata_run-2"
KIT = 1000.0
SECONDS = 20
FIRST = 5
CHANNELS = ["MEG0113", "MEG0112", "MEG0111"]
WORDS = ["<sp>", "if", "you", "<sp>", "go", "a", "<sp>"]
WORD_STARTS = [0.0, 0.1, 0.35, 0.6, 1.3, 1.6, 1.7]
PHONES = ["<sp>", "IH1", "F", "Y", "UW1", "<sp>", "spn", "G", "OW1", "AH0", "<sp>"]
PHONE_STARTS = [0.0, 0.1, 0.2, 0.35, 0.45, 0.6, 0.9, 1.3, 1.4, 1.6, 1.7]
PLAYS = [10.0, 15.0]
OFFSET = 0.02
DRIFT = 0.001
LOCATED = 3.0


def dataset(tmp_path: Path) -> Recording:
    """A downloaded recording with its sidecars, event table and the clip's alignment."""
    fif = tmp_path / "sub-E3b" / "meg" / FIF
    fif.parent.mkdir(parents=True)
    fif.write_bytes(b"fif")
    fif.with_name("sub-E3b_task-en_run-2_meg.json").write_text('{"PowerLineFrequency": 50.0}')
    table = pd.DataFrame({"onset": [1.0, *PLAYS], "duration": 0, "trial_type": [2, 3, 3]})
    table["onset"] += FIRST
    table.to_csv(fif.with_name("sub-E3b_task-en_run-2_events.tsv"), sep="\t", index=False)
    audio = tmp_path / "stimuli" / "audio"
    audio.mkdir(parents=True)
    for kind, items, starts in (("word", WORDS, WORD_STARTS), ("phoneme", PHONES, PHONE_STARTS)):
        (audio / f"mfa_{kind}.txt").write_text("\n".join(items) + "\n")
        (audio / f"mfa_{kind}_stime.txt").write_text("\n".join(str(start) for start in starts) + "\n")
    return Recording(fif)


def synthetic() -> mne.io.RawArray:
    """Three magnetometers, a stim and the audio channel at 1 kHz, the file's first sample `FIRST` seconds in."""
    rng = np.random.default_rng(0)
    data = rng.standard_normal((len(CHANNELS) + 2, round(SECONDS * KIT))) * 1e-13
    names = [*CHANNELS, "STI101", module.AUDIO_CHANNEL]
    info = mne.create_info(names, KIT, ["mag"] * len(CHANNELS) + ["stim", "misc"])
    return mne.io.RawArray(data, info, first_samp=round(FIRST * KIT), verbose="error")


@dataclass
class Calls:
    """What the stubbed reader and SSS were asked."""

    fif: Path | None = None
    sss: dict[str, object] = field(default_factory=dict)


@pytest.fixture
def target() -> Neuromag:
    return Neuromag(mne.create_info(sorted(CHANNELS), KIT, "mag"))


@pytest.fixture
def read(monkeypatch: pytest.MonkeyPatch) -> Calls:
    """`read_raw_fif` answering a synthetic recording, its path recorded; SSS and head-pose fitting stubbed."""
    called = Calls()

    def read_raw_fif(fif: Path, **_kwargs: object) -> mne.io.RawArray:
        called.fif = fif
        return synthetic()

    def maxwell(raw: mne.io.RawArray, **kwargs: object) -> mne.io.RawArray:
        called.sss = kwargs
        return raw

    monkeypatch.setattr(module.mne.io, "read_raw_fif", read_raw_fif)
    monkeypatch.setattr(module, "maxwell_filter", maxwell)
    monkeypatch.setattr(module, "find_bad_channels_maxwell", lambda *_a, **_k: ([CHANNELS[1]], []))
    monkeypatch.setattr(Recording, "head_positions", lambda self, raw: np.zeros((1, 10)))
    return called


@pytest.fixture
def clocked(monkeypatch: pytest.MonkeyPatch) -> None:
    """The audio fit stubbed: every play gets one offset and drift, a run without triggers one play at `LOCATED`."""
    monkeypatch.setattr(Clip, "alignment", Alignment(Envelope(np.zeros(SECONDS))))
    monkeypatch.setattr(Alignment, "clock", lambda _self, _recorded, nominal: Clock(nominal, OFFSET, DRIFT))
    monkeypatch.setattr(Alignment, "locate", lambda _self, _recorded: [LOCATED])


def test_home(tmp_path: Path) -> None:
    root = DataRoot(tmp_path)
    assert Study("a", "b").home(root) == root.raw("a")


def test_tier(tmp_path: Path) -> None:
    clip = Clip(dataset(tmp_path).dataset / "stimuli" / "audio")
    words = clip.tier("word")
    assert words["text"].tolist() == ["", "if", "you", "", "go", "a", ""]
    assert words["end"].tolist() == [*WORD_STARTS[1:], WORD_STARTS[-1]]
    assert clip.tier("phoneme")["text"].tolist()[5:7] == ["", ""]


def test_alignment(tmp_path: Path) -> None:
    clip = Clip(dataset(tmp_path).dataset / "stimuli" / "audio")
    rate = 48000
    audio = np.random.default_rng(0).standard_normal((2 * rate, 2))
    wavfile.write(clip.directory / module.STIMULUS, rate, audio)
    assert len(clip.alignment.reference.samples) == round(2 * KIT)


def test_story(tmp_path: Path) -> None:
    story = Clip(dataset(tmp_path).dataset / "stimuli" / "audio").story
    assert story.words["text"].tolist() == ["if", "you", "go", "a"]
    assert story.words["sentence"].tolist() == [0, 0, 1, 1]
    assert story.phones["text"].tolist() == ["ih_B", "f_E", "y_B", "uw_E", "g_B", "ow_E", "ah_S"]


def test_parsed(tmp_path: Path) -> None:
    assert Recording(tmp_path / FIF).parsed == {"subject": 323, "run": 2}
    assert Recording(tmp_path / "sub-E12_task-en_run-1_meg.fif").parsed == {"subject": 312, "run": 1}
    with pytest.raises(ValueError, match="sub-X1"):
        _ = Recording(tmp_path / "sub-X1_task-en_run-1_meg.fif").parsed


def test_name(tmp_path: Path) -> None:
    assert Recording(tmp_path / FIF).name == NAME


def test_sidecar(tmp_path: Path) -> None:
    rec = Recording(tmp_path / FIF)
    assert rec.sidecar("events.tsv") == tmp_path / "sub-E3b_task-en_run-2_events.tsv"
    assert rec.sidecar("meg.json") == tmp_path / "sub-E3b_task-en_run-2_meg.json"


def test_dataset(tmp_path: Path) -> None:
    assert Recording(tmp_path / "sub-E3b" / "meg" / FIF).dataset == tmp_path


def test_head_positions(monkeypatch: pytest.MonkeyPatch) -> None:
    steps: list[float] = []

    def amplitudes(_raw: object, *, t_step_min: float, **_kwargs: object) -> str:
        steps.append(t_step_min)
        return "amplitudes"

    monkeypatch.setattr(module.mne.chpi, "compute_chpi_amplitudes", amplitudes)
    monkeypatch.setattr(module.mne.chpi, "compute_chpi_locs", lambda _info, got, **_kwargs: got + " locations")
    monkeypatch.setattr(module.mne.chpi, "compute_head_pos", lambda _info, got, **_kwargs: got + " poses")
    assert Recording(Path(FIF)).head_positions(synthetic()) == "amplitudes locations poses"
    assert steps == [module.POSE_STEP]


def test_mains(tmp_path: Path) -> None:
    assert dataset(tmp_path).mains.frequency == 50.0


def test_cleaned(tmp_path: Path, read: Calls) -> None:
    rec = dataset(tmp_path)
    cleaned = rec.cleaned()
    assert read.fif == rec.fif
    assert read.sss["cross_talk"] == tmp_path / module.CROSS_TALK
    assert read.sss["calibration"] == tmp_path / module.CALIBRATION
    assert read.sss["destination"] == module.ORIGIN
    assert cleaned.info["bads"] == [CHANNELS[1]]
    assert cleaned.ch_names == CHANNELS
    assert (len(cleaned.ch_names), len(cleaned.times)) == (len(CHANNELS), SECONDS * SFREQ)
    assert cleaned.info["highpass"] == HIGHPASS


@pytest.mark.usefixtures("read")
def test_written(tmp_path: Path, target: Neuromag) -> None:
    signal, attrs = dataset(tmp_path).written(target)
    assert signal.shape == (len(CHANNELS), SECONDS * SFREQ)
    assert signal.dtype == np.float32
    assert attrs["bad_channels"] == CHANNELS[1]
    assert attrs["mains"] == 50.0
    assert attrs["channel_names"] == ", ".join(sorted(CHANNELS))


@pytest.mark.usefixtures("read")
def test_clip(tmp_path: Path) -> None:
    assert dataset(tmp_path).clip == Clip(tmp_path / "stimuli" / "audio")


@pytest.mark.usefixtures("read")
def test_triggers(tmp_path: Path) -> None:
    assert dataset(tmp_path).triggers() == pytest.approx(PLAYS)


@pytest.mark.usefixtures("read")
def test_audio(tmp_path: Path) -> None:
    audio = dataset(tmp_path).audio()
    assert len(audio.samples) == SECONDS * round(KIT)
    assert audio.samples.mean() == pytest.approx(0, abs=1e-9)


@pytest.mark.usefixtures("read", "clocked")
def test_plays(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rec = dataset(tmp_path)
    assert rec.plays() == [Clock(play, OFFSET, DRIFT) for play in PLAYS]
    monkeypatch.setattr(Recording, "triggers", lambda _self: [])
    assert rec.plays() == [Clock(LOCATED, OFFSET, DRIFT)]


@pytest.mark.usefixtures("read", "clocked")
def test_events(tmp_path: Path) -> None:
    events = dataset(tmp_path).events()
    assert list(events.columns) == COLUMNS
    assert len(events) == 2 * (4 + 7)
    assert events["timemeg"].is_monotonic_increasing
    words = events[events["kind"] == "word"]
    assert words["sentenceidx"].tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    first = [play + OFFSET + (1 + DRIFT) * 0.1 for play in PLAYS]
    assert words["timemeg"].tolist()[0:5:4] == pytest.approx(first)
    assert words["duration"].iloc[0] == pytest.approx((1 + DRIFT) * 0.25)


@pytest.mark.usefixtures("read")
def test_events_at(tmp_path: Path) -> None:
    events = dataset(tmp_path).events_at([Clock(PLAYS[0], 0.0, 0.0)])
    assert len(events) == 4 + 7
    assert events["timemeg"].iloc[0] == pytest.approx(PLAYS[0] + 0.1)


def test_under(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_OVERRIDE, str(tmp_path))
    root = DataRoot.from_config()
    assert Kymata.under(root) == Kymata(KYMATA.home(root), root.processed(KYMATA.dataset))


def test_recordings(tmp_path: Path) -> None:
    ready = dataset(tmp_path)
    unfinished = tmp_path / "sub-E4" / "meg" / "sub-E4_task-en_run-1_meg.fif"
    unfinished.parent.mkdir(parents=True)
    unfinished.touch()
    study = Kymata(tmp_path, tmp_path / "out")
    assert list(study.recordings()) == [ready]
    assert list(study.recordings((323,))) == [ready]
    assert list(study.recordings((304,))) == []


@pytest.mark.usefixtures("read", "clocked")
def test_convert(tmp_path: Path, target: Neuromag) -> None:
    rec, study = dataset(tmp_path / "in"), Kymata(tmp_path / "in", tmp_path / "out")
    meg = study.convert(rec, target)
    assert meg == tmp_path / "out" / "serialised" / f"{NAME}_meg.h5"
    assert study.events(rec).exists()


def test_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: Neuromag) -> None:
    monkeypatch.setenv(ENV_OVERRIDE, str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["kymata", "--workers", "1", "--target", str(tmp_path / "scratch")])
    rec = dataset(Kymata.under(DataRoot.from_config()).source)
    converted: list[Recording] = []
    monkeypatch.setattr(module, "ProcessPoolExecutor", ThreadPoolExecutor)
    monkeypatch.setattr(Neuromag, "of", classmethod(lambda cls, path: target))
    monkeypatch.setattr(Kymata, "convert", lambda self, recording, _target: converted.append(recording))
    main()
    assert converted == [rec]


def converted(study: Kymata, rec: Recording) -> None:
    """`rec` marked converted: an h5 stand-in and the events directory."""
    study.meg(rec).parent.mkdir(parents=True)
    study.meg(rec).touch()
    study.events(rec).parent.mkdir(parents=True)


@pytest.mark.usefixtures("read", "clocked")
def test_refresh(tmp_path: Path) -> None:
    rec, study = dataset(tmp_path / "in"), Kymata(tmp_path / "in", tmp_path / "out")
    assert study.refresh(rec) == []
    converted(study, rec)
    assert study.refresh(rec) == [Clock(play, OFFSET, DRIFT) for play in PLAYS]
    assert len(pd.read_csv(study.events(rec), sep="\t")) == 2 * (4 + 7)
    assert study.meg(rec).read_bytes() == b""


@pytest.mark.usefixtures("read", "clocked")
def test_refreshed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    good, study = dataset(tmp_path / "in"), Kymata(tmp_path / "in", tmp_path / "out")
    bad = Recording(tmp_path / "sub-E5_task-en_run-1_meg.fif")
    monkeypatch.setattr(module, "ProcessPoolExecutor", ThreadPoolExecutor)
    converted(study, good)
    study.meg(bad).touch()
    assert study.refreshed([bad, good], 2) == [Clock(play, OFFSET, DRIFT) for play in PLAYS]
    assert study.events(good).exists()


def test_main_events_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: Neuromag) -> None:
    monkeypatch.setenv(ENV_OVERRIDE, str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["kymata", "--events-only"])
    rec = dataset(Kymata.under(DataRoot.from_config()).source)
    seen: list[tuple[list[Recording], int]] = []

    def refreshed(_self: Kymata, recordings: list[Recording], workers: int) -> list[Clock]:
        seen.append((recordings, workers))
        return [Clock(1.0, OFFSET, DRIFT)]

    monkeypatch.setattr(Neuromag, "of", classmethod(lambda cls, path: target))
    monkeypatch.setattr(Kymata, "refreshed", refreshed)
    main()
    assert seen == [([rec], module.EVENT_WORKERS)]


def test_shared(tmp_path: Path) -> None:
    rec = dataset(tmp_path)
    assert rec.shared == {
        "cross_talk": rec.dataset / module.CROSS_TALK,
        "calibration": rec.dataset / module.CALIBRATION,
    }


def test_audio_channel(tmp_path: Path) -> None:
    assert dataset(tmp_path).audio_channel == module.AUDIO_CHANNEL
