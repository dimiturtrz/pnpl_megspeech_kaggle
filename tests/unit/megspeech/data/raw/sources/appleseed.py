import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import cast

import mne
import numpy as np
import pandas as pd
import pytest

from core.paths import ENV_OVERRIDE, DataRoot
from megspeech.data.constants import SFREQ
from megspeech.data.raw.fetch import APPLESEED
from megspeech.data.raw.sources import appleseed as module
from megspeech.data.raw.sources.appleseed import Appleseed, Bads, Recording, Reference, Story, main
from megspeech.data.raw.sources.masc import COLUMNS, HIGHPASS, Neuromag
from megspeech.data.raw.textgrid import TextGrid

FIF = "sub-08_task-Appleseed_split-01_meg.fif"
NAME = "sub-208_ses-1_task-Appleseed_run-1"
KIT = 1000.0
SECONDS = 40
MEG, REF = 20, 2
FLAT, NOISY = 3, 7
WORDS = [(0.0, 0.25, "if"), (0.25, 0.5, "you"), (0.5, 1.2, ""), (1.2, 1.5, "go"), (1.5, 1.6, "a")]
PHONES = [
    (0.0, 0.1, "IH1"),
    (0.1, 0.25, "F"),
    (0.25, 0.35, "Y"),
    (0.35, 0.5, "UW1"),
    (0.5, 1.2, "sil"),
    (1.2, 1.3, "G"),
    (1.3, 1.5, "OW1"),
    (1.5, 1.6, "AH0"),
]


def tier(name: str, intervals: list[tuple[float, float, str]]) -> str:
    rows = "".join(
        f'\t\t\tintervals [{i}]:\n\t\t\t\txmin = {start}\n\t\t\t\txmax = {end}\n\t\t\t\ttext = "{text}"\n'
        for i, (start, end, text) in enumerate(intervals, 1)
    )
    return f'\t\tclass = "IntervalTier"\n\t\tname = "{name}"\n\t\tintervals: size = {len(intervals)}\n{rows}'


GRID = "item []:\n\titem [1]:\n" + tier("words", WORDS) + "\titem [2]:\n" + tier("phones", PHONES)


def story(tmp_path: Path) -> Story:
    path = tmp_path / "1.TextGrid"
    path.write_text(GRID, encoding="utf-8")
    return Story(TextGrid.read(path))


def recording(tmp_path: Path) -> Recording:
    """A downloaded recording: its first split, sidecar, events and its two stories' TextGrids."""
    fif = tmp_path / "sub-08" / "meg" / FIF
    fif.parent.mkdir(parents=True)
    fif.write_bytes(b"fif")
    fif.with_name("sub-08_task-Appleseed_meg.json").write_text('{"PowerLineFrequency": 50.0}')
    table = pd.DataFrame(
        {
            "onset": [10.0, 20.0, 30.0, 40.0, 50.0],
            "trial_type": ["onset", "offset", "onset", "offset", "onset"],
            "stimulus": ["1", "n/a", "11b", "n/a", "n/a"],
        }
    )
    table.to_csv(fif.with_name("sub-08_task-Appleseed_events.tsv"), sep="\t", index=False)
    (tmp_path / "stimuli").mkdir()
    for stimulus in ("1", "11b"):
        (tmp_path / "stimuli" / f"{stimulus}.TextGrid").write_text(GRID, encoding="utf-8")
    return Recording(fif)


def synthetic() -> mne.io.RawArray:
    """20 MEG channels and 2 reference channels sharing one environment; one MEG channel flat, one noisy."""
    rng = np.random.default_rng(0)
    environment = rng.standard_normal((1, round(SECONDS * KIT))) * 1e-12
    gains = np.linspace(0.95, 1.05, MEG)[:, None]
    meg = gains * environment + rng.standard_normal((MEG, environment.shape[1])) * 1e-14
    meg[FLAT] = 0.0
    meg[NOISY] *= 1000
    reference = np.vstack([environment, 0.3 * environment + rng.standard_normal(environment.shape) * 1e-14])
    names = [f"MEG {number:03d}" for number in range(MEG + REF)]
    info = mne.create_info(names, KIT, ["mag"] * MEG + ["ref_meg"] * REF)
    return mne.io.RawArray(np.vstack([meg, reference]), info, verbose="error")


@pytest.fixture
def target() -> Neuromag:
    return Neuromag(mne.create_info(["MEG 0111", "MEG 0121", "MEG 0131"], KIT, "mag"))


@pytest.fixture
def read(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """`read_raw_fif` answering a synthetic recording, its path recorded."""
    called: dict[str, object] = {}

    def read_raw_fif(fif: Path, **_kwargs: object) -> mne.io.RawArray:
        called["fif"] = fif
        return synthetic()

    monkeypatch.setattr(module.mne.io, "read_raw_fif", read_raw_fif)
    return called


@pytest.fixture
def mapped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Neuromag, "mapping", lambda self, source: np.ones((len(self.names), len(source["ch_names"]))))


def test_limit() -> None:
    assert Bads.limit(157) == pytest.approx(2.95, abs=0.02)
    assert Bads.limit(20) < Bads.limit(157)


def test_find() -> None:
    rng = np.random.default_rng(0)
    meg = rng.standard_normal((MEG, 500)) * np.linspace(0.95, 1.05, MEG)[:, None]
    meg[FLAT] = 0.0
    meg[NOISY] *= 1000
    assert np.flatnonzero(Bads.find(meg).flagged).tolist() == [FLAT, NOISY]


def test_clean() -> None:
    rng = np.random.default_rng(0)
    environment = rng.standard_normal((2, 4000))
    meg = np.array([[2.0, 1.0], [-1.0, 0.5]]) @ environment + rng.standard_normal((2, 4000)) * 0.01
    cleaned, removed = Reference(environment).clean(meg)
    assert removed > 0.999
    assert cleaned.std() < 0.02
    unrelated, none = Reference(rng.standard_normal((2, 4000))).clean(rng.standard_normal((2, 4000)))
    assert none < 0.01
    assert unrelated.shape == (2, 4000)


def test_words(tmp_path: Path) -> None:
    words = story(tmp_path).words
    assert words["text"].tolist() == ["if", "you", "go", "a"]
    assert words["new"].tolist() == [True, False, True, False]
    assert words["sentence"].tolist() == [0, 0, 1, 1]
    assert words["wordidx"].tolist() == [0, 1, 0, 1]


def test_phones(tmp_path: Path) -> None:
    phones = story(tmp_path).phones
    assert phones["text"].tolist() == ["ih_B", "f_E", "y_B", "uw_E", "g_B", "ow_E", "ah_S"]
    assert phones["place"].tolist() == [0, 1, 0, 1, 0, 1, 0]
    assert phones["word_sentence"].tolist() == [0, 0, 0, 0, 1, 1, 1]


def test_story_events(tmp_path: Path) -> None:
    events = story(tmp_path).events(10.0)
    assert len(events) == 4 + 7
    words = events[events["kind"] == "word"]
    assert words["timemeg"].tolist() == pytest.approx([10.0, 10.25, 11.2, 11.5])
    assert words["duration"].tolist() == pytest.approx([0.25, 0.25, 0.3, 0.1])
    assert words["phonemeidx"].isna().all()
    phones = events[events["kind"] == "phoneme"]
    assert phones["wordidx"].tolist() == [0, 0, 1, 1, 0, 0, 1]
    assert phones["timemeg"].iloc[1] == pytest.approx(10.1)


def test_parsed(tmp_path: Path) -> None:
    assert Recording(tmp_path / FIF).parsed == {"subject": 8}
    assert Recording(tmp_path / "sub-08_task-Appleseed_meg.fif").parsed == {"subject": 8}
    with pytest.raises(ValueError, match="split-02"):
        _ = Recording(tmp_path / "sub-08_task-Appleseed_split-02_meg.fif").parsed


def test_name(tmp_path: Path) -> None:
    assert Recording(tmp_path / FIF).name == NAME


def test_sidecar(tmp_path: Path) -> None:
    rec = Recording(tmp_path / FIF)
    assert rec.sidecar("events.tsv") == tmp_path / "sub-08_task-Appleseed_events.tsv"
    assert rec.sidecar("meg.json") == tmp_path / "sub-08_task-Appleseed_meg.json"


def test_stimuli(tmp_path: Path) -> None:
    assert Recording(tmp_path / "sub-08" / "meg" / FIF).stimuli == tmp_path / "stimuli"


def test_cleaned(tmp_path: Path, read: dict[str, object]) -> None:
    rec = recording(tmp_path)
    cleaned = rec.cleaned()
    assert read["fif"] == rec.fif
    assert cleaned.bads == [f"MEG {FLAT:03d}", f"MEG {NOISY:03d}"]
    assert cleaned.meg.shape == (MEG - 2, SECONDS * SFREQ)
    assert cleaned.info["sfreq"] == SFREQ
    assert cleaned.info["highpass"] == HIGHPASS
    assert cleaned.info.ch_names == [f"MEG {n:03d}" for n in range(MEG) if n not in (FLAT, NOISY)]
    assert cleaned.removed > 0.9


@pytest.mark.usefixtures("read", "mapped")
def test_written(tmp_path: Path, target: Neuromag) -> None:
    signal, attrs = recording(tmp_path).written(target)
    assert signal.shape == (3, SECONDS * SFREQ)
    assert signal.dtype == np.float32
    assert attrs["bad_channels"] == f"MEG {FLAT:03d}, MEG {NOISY:03d}"
    assert attrs["mains"] == 50.0
    assert attrs["channel_names"] == "MEG0111, MEG0121, MEG0131"
    assert float(cast("float", attrs["reference_variance_removed"])) > 0.9


def test_recording_events(tmp_path: Path) -> None:
    events = recording(tmp_path).events()
    assert list(events.columns) == COLUMNS
    assert len(events) == 2 * 11
    assert events["kind"].tolist()[:3] == ["word", "phoneme", "phoneme"]
    assert events["timemeg"].is_monotonic_increasing
    assert events[events["kind"] == "word"]["sentenceidx"].tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    assert events["timemeg"].iloc[-1] == pytest.approx(31.5)


def test_under(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_OVERRIDE, str(tmp_path))
    root = DataRoot.from_config()
    assert Appleseed.under(root) == Appleseed(APPLESEED.home(root), root.processed(APPLESEED.dataset))


def test_recordings(tmp_path: Path) -> None:
    ready = recording(tmp_path)
    ready.fif.with_name("sub-08_task-Appleseed_split-02_meg.fif").write_bytes(b"fif")
    ready.fif.with_name("sub-08_task-Tone_meg.fif").write_bytes(b"fif")
    unfinished = tmp_path / "sub-09" / "meg" / "sub-09_task-Appleseed_split-01_meg.fif"
    unfinished.parent.mkdir(parents=True)
    unfinished.touch()
    study = Appleseed(tmp_path, tmp_path / "out")
    assert list(study.recordings()) == [ready]
    assert list(study.recordings((8,))) == [ready]
    assert list(study.recordings((7,))) == []


@pytest.mark.usefixtures("read", "mapped")
def test_convert(tmp_path: Path, target: Neuromag) -> None:
    rec, study = recording(tmp_path / "in"), Appleseed(tmp_path / "in", tmp_path / "out")
    meg = study.convert(rec, target)
    assert meg == tmp_path / "out" / "serialised" / f"{NAME}_meg.h5"
    assert study.events(rec).exists()


def test_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: Neuromag) -> None:
    monkeypatch.setenv(ENV_OVERRIDE, str(tmp_path))
    out = tmp_path / "scratch"
    monkeypatch.setattr(sys, "argv", ["appleseed", "--workers", "1", "--target", str(out)])
    rec = recording(Appleseed.under(DataRoot.from_config()).source)
    converted: list[Recording] = []
    monkeypatch.setattr(module, "ProcessPoolExecutor", ThreadPoolExecutor)
    monkeypatch.setattr(Neuromag, "of", classmethod(lambda cls, path: target))
    monkeypatch.setattr(Appleseed, "convert", lambda self, recording, _target: converted.append(recording))
    main()
    assert converted == [rec]
