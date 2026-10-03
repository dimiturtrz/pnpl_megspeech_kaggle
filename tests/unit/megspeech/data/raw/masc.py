import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import mne
import numpy as np
import pandas as pd
import pytest

from core.paths import ENV_OVERRIDE, DataRoot
from megspeech.data.constants import SFREQ
from megspeech.data.raw import masc as module
from megspeech.data.raw.masc import COLUMNS, HIGHPASS, Masc, Neuromag, Recording, main
from megspeech.data.raw.runs import Run

CON = "sub-01_ses-0_task-2_meg.con"
NAME = "sub-101_ses-0_task-MASC2_run-1"
KIT = 1000.0
SECONDS = 40
TRIALS = [
    {"kind": "sound", "sound": "a.wav"},
    {"kind": "word", "word": "the", "sequence_id": 3, "condition": "sentence", "word_index": 0},
    {"kind": "phoneme", "word": "the", "phoneme": "dh_B", "sequence_id": 3, "condition": "sentence", "word_index": 0},
    {"kind": "phoneme", "word": "the", "phoneme": "ah_E", "sequence_id": 3, "condition": "sentence", "word_index": 0},
    {"kind": "word", "word": "cat", "sequence_id": 4, "condition": "word_list", "word_index": 1},
]


def info(names: list[str]) -> mne.Info:
    return mne.create_info(names, KIT, "mag")


def kit(names: tuple[str, ...] = ("MEG 001", "MEG 002")) -> mne.io.RawArray:
    signal = np.random.default_rng(0).standard_normal((len(names), round(SECONDS * KIT))) * 1e-12
    return mne.io.RawArray(signal, info(list(names)), verbose="error")


def recording(tmp_path: Path) -> Recording:
    """A downloaded recording: its `.con`, markers, events and its session's head shapes."""
    con = tmp_path / "sub-01" / "ses-0" / "meg" / CON
    con.parent.mkdir(parents=True)
    con.write_bytes(b"kit")
    con.with_name(CON.replace("meg.con", "markers.mrk")).touch()
    for shape in ("ELP", "HSP"):
        con.with_name(f"sub-01_ses-0_acq-{shape}_headshape.pos").write_text("% header 18:34:26\n1 2 3\n4 5 6\n")
    table = pd.DataFrame(
        {"onset": [0.5, 1.0, 1.0, 1.1, 2.0], "duration": [0.3, 0.2, 0.1, 0.1, 0.25], "trial_type": map(str, TRIALS)}
    )
    table.to_csv(con.with_name(CON.replace("meg.con", "events.tsv")), sep="\t", index=False)
    return Recording(con)


@pytest.fixture
def target(tmp_path: Path) -> Neuromag:
    return Neuromag(info(["MEG 0111", "MEG 0121", "MEG 0131"]))


@pytest.fixture
def read(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """`read_raw_kit` answering a synthetic recording, its arguments recorded."""
    called: dict[str, object] = {}

    def read_raw_kit(con: Path, **kwargs: object) -> mne.io.RawArray:
        called.update(kwargs, con=con)
        return kit()

    monkeypatch.setattr(module.mne.io, "read_raw_kit", read_raw_kit)
    return called


@pytest.fixture
def mapped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Neuromag, "mapping", lambda self, source: np.ones((len(self.names), len(source["ch_names"]))))


def test_of(tmp_path: Path) -> None:
    path = tmp_path / "neuromag_raw.fif"
    channels = mne.create_info(["MEG 0121", "STI 014", "MEG 0111"], KIT, ["mag", "stim", "grad"])
    raw = mne.io.RawArray(np.zeros((3, 10)), channels)
    raw.save(path, verbose="error")
    assert Neuromag.of(path).info["ch_names"] == ["MEG 0111", "MEG 0121"]


def test_names(target: Neuromag) -> None:
    assert target.names == ["MEG0111", "MEG0121", "MEG0131"]


def test_types(target: Neuromag) -> None:
    assert target.types == ["mag"] * 3


def test_mapping(target: Neuromag, monkeypatch: pytest.MonkeyPatch) -> None:
    source = info(["MEG 001"])
    called: list[tuple[object, ...]] = []
    monkeypatch.setattr(module, "_map_meg_or_eeg_channels", lambda *args, **kwargs: called.append((*args, kwargs)))
    target.mapping(source)
    assert called == [(source, target.info, {"mode": "accurate", "origin": module.ORIGIN})]


def test_parsed(tmp_path: Path) -> None:
    assert Recording(tmp_path / CON).parsed == {"subject": 1, "session": 0, "task": 2}
    with pytest.raises(ValueError, match="other"):
        _ = Recording(tmp_path / "other.con").parsed


def test_name(tmp_path: Path) -> None:
    assert Recording(tmp_path / CON).name == NAME
    assert Run(tmp_path / f"{NAME}_meg.h5").subject == 101


def test_sidecar(tmp_path: Path) -> None:
    rec = Recording(tmp_path / CON)
    assert rec.sidecar("events.tsv") == tmp_path / "sub-01_ses-0_task-2_events.tsv"
    assert rec.sidecar("acq-ELP_headshape.pos", session=True) == tmp_path / "sub-01_ses-0_acq-ELP_headshape.pos"


def test_points(tmp_path: Path) -> None:
    path = recording(tmp_path).sidecar("acq-HSP_headshape.pos", session=True)
    assert np.allclose(Recording.points(path), [[1e-3, 2e-3, 3e-3], [4e-3, 5e-3, 6e-3]])


def test_raw(tmp_path: Path, read: dict[str, object]) -> None:
    rec = recording(tmp_path)
    raw = rec.raw()
    assert raw.info["sfreq"] == SFREQ
    assert raw.info["highpass"] == HIGHPASS
    assert raw.n_times == SECONDS * SFREQ
    assert read["con"] == rec.con
    assert read["mrk"] == rec.sidecar("markers.mrk")
    elp = read["elp"]
    assert isinstance(elp, np.ndarray)
    assert elp.shape == (2, 3)


@pytest.mark.usefixtures("read", "mapped")
def test_recording_meg(tmp_path: Path, target: Neuromag) -> None:
    signal = recording(tmp_path).meg(target)
    assert signal.shape == (3, SECONDS * SFREQ)
    assert signal.dtype == np.float32


def test_masc_meg(tmp_path: Path) -> None:
    rec = Recording(tmp_path / CON)
    assert Masc(tmp_path, tmp_path / "out").meg(rec) == tmp_path / "out" / "serialised" / f"{NAME}_meg.h5"


def test_masc_events(tmp_path: Path) -> None:
    rec = Recording(tmp_path / CON)
    assert Masc(tmp_path, tmp_path / "out").events(rec) == tmp_path / "out" / "events" / f"{NAME}_events.tsv"


def test_recording_events(tmp_path: Path) -> None:
    rec = recording(tmp_path)
    events = rec.events()
    assert list(events.columns) == COLUMNS
    assert events["kind"].tolist() == ["word", "phoneme", "phoneme", "word"]
    assert events["segment"].tolist() == ["the", "dh_B", "ah_E", "cat"]
    assert events["sentenceidx"].tolist()[:3] == [3, 3, 3]
    assert np.isnan(events["sentenceidx"].iloc[3])
    assert events["phonemeidx"].tolist()[1:3] == [0, 1]
    assert events["timemeg"].tolist() == [1.0, 1.0, 1.1, 2.0]


def test_under(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_OVERRIDE, str(tmp_path))
    root = DataRoot.from_config()
    assert Masc.under(root) == Masc(root.raw("meg_masc"), root.processed("meg_masc"))


def test_recordings(tmp_path: Path) -> None:
    ready = recording(tmp_path)
    ready.con.with_name("sub-01_ses-0_task-3_meg.con").touch()
    ready.con.with_name("sub-01_ses-0_task-1_meg.con").write_bytes(b"kit")
    masc = Masc(tmp_path, tmp_path / "out")
    assert list(masc.recordings()) == [ready]
    assert list(masc.recordings((1,))) == [ready]
    assert list(masc.recordings((7,))) == []


@pytest.mark.usefixtures("read", "mapped")
def test_convert(tmp_path: Path, target: Neuromag) -> None:
    rec, masc = recording(tmp_path), Masc(tmp_path / "in", tmp_path / "out")
    meg = masc.convert(rec, target)
    with h5py.File(meg) as stored:
        assert stored["data"].shape == (3, SECONDS * SFREQ)
        assert stored.attrs["sample_frequency"] == SFREQ
        assert stored.attrs["channel_names"] == "MEG0111, MEG0121, MEG0131"
    [run] = Run.discover(masc.target)
    assert run.name == NAME
    meg.unlink()
    assert masc.convert(rec, target) == meg
    assert not meg.exists()


def test_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: Neuromag) -> None:
    monkeypatch.setenv(ENV_OVERRIDE, str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["masc", "--workers", "1"])
    rec = recording(Masc.under(DataRoot.from_config()).source)
    converted: list[Recording] = []
    monkeypatch.setattr(module, "ProcessPoolExecutor", ThreadPoolExecutor)
    monkeypatch.setattr(Neuromag, "of", classmethod(lambda cls, path: target))
    monkeypatch.setattr(Masc, "convert", lambda self, recording, _target: converted.append(recording))
    main()
    assert converted == [rec]
