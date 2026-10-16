import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import pytest
from scipy.io import savemat, wavfile

from core.paths import ENV_OVERRIDE, DataRoot
from megspeech.data.constants import SFREQ
from megspeech.data.labels.vocabulary import ABSENT, PRIMARY
from megspeech.data.raw.sources import smn4lang as module
from megspeech.data.raw.sources.kymata_soto_sync import Clock
from megspeech.data.raw.sources.meg_masc import HIGHPASS, Neuromag
from megspeech.data.raw.sources.smn4lang import MANDARIN, SMN4LANG, Narration, Punctuated, Recording, Smn4Lang, main
from megspeech.data.raw.textgrid import TextGrid

FIF = "sub-03_task-RDR_run-7_meg.fif"
NAME = "sub-503_ses-1_task-SMN4Lang_run-7"
RATE = 16000
SECONDS = 4.0
START, LATE = 2.5, 9.0


def dataset(tmp_path: Path, start: float = START) -> Recording:
    """A downloaded recording, its events starting the story at `start`, and the story's wav and word alignment."""
    fif = tmp_path / "sub-03" / "meg" / FIF
    fif.parent.mkdir(parents=True)
    fif.write_bytes(b"fif")
    end = START + SECONDS
    fif.with_name("sub-03_task-RDR_run-7_events.tsv").write_text(
        f"onset\tduration\ttrial_type\tvalue\tsample\n{start}\t0\tBeg\t1\t0\n{end}\t0\tEnd\t2\t0\n"
    )
    audio = tmp_path / "stimuli" / "audio"
    audio.mkdir(parents=True)
    wavfile.write(audio / "story_7.wav", RATE, np.zeros(int(SECONDS * RATE), dtype=np.int16))
    aligned = tmp_path / module.WORD_TIMES
    aligned.mkdir(parents=True)
    shift = module.ALIGNMENT_SHIFT
    savemat(
        aligned / "story_7_word_time.mat",
        {
            "start": np.array([[0.5, 0.8, 1.2, 1.3, 2.5]]) + shift,
            "end": np.array([[0.8, 1.2, 1.2, 2.0, 3.0]]) + shift,
            "word": np.array(["我们 ", "在  ", "。  ", "中国", "，  "]),
        },
    )
    return Recording(fif)


def test_dataset(tmp_path: Path) -> None:
    assert dataset(tmp_path).book.dataset == tmp_path


def test_narration_words(tmp_path: Path) -> None:
    words = dataset(tmp_path).book.words
    assert words["segment"].tolist() == ["我们", "在", "中国"]
    assert words["timemeg"].tolist() == pytest.approx([0.5, 0.8, 1.3])
    assert words["sentenceidx"].tolist() == [0, 0, 1]
    assert words["wordidx"].tolist() == [0, 1, 0]


def test_punctuated_words() -> None:
    tier = pd.DataFrame({"start": [0.0, 0.3, 0.3, 0.4], "end": [0.3, 0.3, 0.3, 0.6], "text": ["好", "！", "“", "走"]})
    words = Punctuated(TextGrid.aligned(tier, tier.iloc[:0])).words
    assert words["text"].tolist() == ["好", "走"]
    assert words["sentence"].tolist() == [0, 1]
    assert words["new"].tolist() == [True, True]


def test_mark() -> None:
    assert Punctuated.mark("，")
    assert Punctuated.mark("“")
    assert Punctuated.mark("")
    assert not Punctuated.mark("中国")


def test_duration(tmp_path: Path) -> None:
    assert dataset(tmp_path).book.duration == SECONDS


def test_parsed(tmp_path: Path) -> None:
    assert Recording(tmp_path / FIF).parsed == {"subject": 503, "run": 7}
    with pytest.raises(ValueError, match="sub-X1"):
        _ = Recording(tmp_path / "sub-X1_task-RDR_run-1_meg.fif").parsed


def test_name(tmp_path: Path) -> None:
    assert Recording(tmp_path / FIF).name == NAME


def test_book(tmp_path: Path) -> None:
    assert dataset(tmp_path).book == Narration(tmp_path / "stimuli" / "audio" / "story_7")


def test_prefix(tmp_path: Path) -> None:
    assert Recording(tmp_path / FIF).prefix == MANDARIN


def test_mains(tmp_path: Path) -> None:
    assert Recording(tmp_path / FIF).mains.frequency == module.MAINS.frequency


def test_cleaned(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The saved recording is not Maxwell-filtered again: only its MEG channels kept, notched, high-passed, 250 Hz."""
    info = mne.create_info(["MEG0111", "MEG0112", "STI101"], 1000.0, ["mag", "grad", "stim"])
    noise = np.random.default_rng(0).standard_normal((3, 2000)) * 1e-12
    monkeypatch.setattr(mne.io, "read_raw_fif", lambda *_args, **_kwargs: mne.io.RawArray(noise, info, verbose="error"))
    cleaned = Recording(tmp_path / FIF).cleaned()
    assert cleaned.ch_names == ["MEG0111", "MEG0112"]
    assert cleaned.info["sfreq"] == SFREQ
    assert cleaned.info["highpass"] == HIGHPASS


def test_marks(tmp_path: Path) -> None:
    assert dataset(tmp_path).marks() == {"Beg": START, "End": START + SECONDS}


def test_plays(tmp_path: Path) -> None:
    """The start trigger when the span matches the story; else the end trigger less the story."""
    assert dataset(tmp_path / "good").plays() == [Clock(START, 0.0, 0.0)]
    assert dataset(tmp_path / "broken", LATE).plays() == [Clock(START, 0.0, 0.0)]
    assert LATE - START > module.SPAN_TOLERANCE


def test_events_at(tmp_path: Path) -> None:
    events = dataset(tmp_path).events_at([Clock(10.0, 0.0, 0.0)])
    assert events["segment"].tolist() == [MANDARIN + word for word in ("我们", "在", "中国")]
    assert events["timemeg"].tolist() == pytest.approx([10.5, 10.8, 11.3])
    assert events["timechapter"].tolist() == pytest.approx([0.5, 0.8, 1.3])


def test_events_out_of_vocabulary(tmp_path: Path) -> None:
    segments = dataset(tmp_path).events_at([Clock(0.0, 0.0, 0.0)])["segment"]
    assert (PRIMARY.labels(segments) == ABSENT).all()


def test_under(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_OVERRIDE, str(tmp_path))
    root = DataRoot.from_config()
    assert Smn4Lang.under(root) == Smn4Lang(SMN4LANG.home(root), root.processed(SMN4LANG.dataset))


def test_recordings(tmp_path: Path) -> None:
    ready = dataset(tmp_path)
    unfinished = tmp_path / "sub-04" / "meg" / "sub-04_task-RDR_run-1_meg.fif"
    unfinished.parent.mkdir(parents=True)
    unfinished.touch()
    study = Smn4Lang(tmp_path, tmp_path / "out")
    assert list(study.recordings()) == [ready]
    assert list(study.recordings((503,))) == [ready]
    assert list(study.recordings((504,))) == []


def test_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_OVERRIDE, str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["smn4lang", "--workers", "1"])
    rec = dataset(Smn4Lang.under(DataRoot.from_config()).source)
    converted: list[Recording] = []
    target = Neuromag(mne.create_info(["MEG0111"], 1000.0, "mag"))
    monkeypatch.setattr(module, "ProcessPoolExecutor", ThreadPoolExecutor)
    monkeypatch.setattr(Neuromag, "of", classmethod(lambda cls, path: target))
    monkeypatch.setattr(Smn4Lang, "convert", lambda self, recording, _target: converted.append(recording))
    main()
    assert converted == [rec]
