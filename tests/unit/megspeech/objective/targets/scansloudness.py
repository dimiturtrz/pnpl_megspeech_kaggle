from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.io import wavfile

from core.paths import ENV_OVERRIDE, DataRoot
from megspeech.data.raw.runs import Run
from megspeech.data.raw.scans import SCANS, STIMULI
from megspeech.data.raw.smn4lang import SMN4LANG, STORIES
from megspeech.objective.targets.loudness import CUT, LENGTH, Loudness
from megspeech.objective.targets.scansloudness import Audiobook, Narrated
from megspeech.objective.targets.speech import SAMPLING
from tests.unit.megspeech.data.raw.runs import meg
from tests.unit.megspeech.objective.targets.speech import SECONDS, loud

NAME = "sub-401_ses-1_task-SCANSab2_run-1"
STORY = "sub-501_ses-1_task-SMN4Lang_run-7"


def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DataRoot:
    monkeypatch.setenv(ENV_OVERRIDE, str(tmp_path))
    return DataRoot.from_config()


def served(data: DataRoot, run: Run | None = None, wav: Path | None = None) -> Run:
    """A converted run whose words start at 0.94 s, 2 s and 3.8 s of the wav it played, silent until 1 s: a
    MEG-SCANS part unless another `run` and its `wav` are given."""
    run = run or Run(meg(data.processed(SCANS.dataset), NAME))
    wav = wav or SCANS.home(data) / STIMULI / "task-audiobook2_run-01_stim.wav"
    run.events.parent.mkdir(parents=True, exist_ok=True)
    events = pd.DataFrame(
        {"kind": ["word", "word", "word"], "segment": ["de:a", "de:b", "de:c"], "timechapter": [0.94, 2.0, 3.8]}
    )
    events.to_csv(run.events, sep="\t", index=False)
    wav.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(wav, SAMPLING, (loud(SECONDS, 1.0).audio * 1000).astype(np.int16))
    return run


def story(data: DataRoot) -> Run:
    """A converted SMN4Lang run of story 7, served as a MEG-SCANS part is."""
    run = Run(meg(data.processed(SMN4LANG.dataset), STORY))
    return served(data, run, SMN4LANG.home(data) / STORIES / "story_7.wav")


def test_audiobook_wav(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = root(tmp_path, monkeypatch)
    assert (
        Audiobook(Run(meg(tmp_path, NAME))).wav(data) == SCANS.home(data) / STIMULI / "task-audiobook2_run-01_stim.wav"
    )


def test_narrated_wav(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = root(tmp_path, monkeypatch)
    assert Narrated(Run(meg(tmp_path, STORY))).wav(data) == SMN4LANG.home(data) / STORIES / "story_7.wav"


def test_targets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = root(tmp_path, monkeypatch)
    targets = Audiobook(served(data)).targets(data, CUT)
    assert targets.vectors.shape == (3, LENGTH)
    assert targets.present[0]


def test_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = root(tmp_path, monkeypatch)
    served(data)
    directory = Audiobook.write(data, CUT)
    assert directory == Loudness.directory(data, CUT)
    assert [path.stem for path in directory.glob("*.npy")] == [NAME]
    assert Audiobook.write(data, CUT) == directory
    story(data)
    assert Narrated.write(data, CUT) == directory
    assert sorted(path.stem for path in directory.glob("*.npy")) == [NAME, STORY]


def test_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = root(tmp_path, monkeypatch)
    served(data)
    story(data)
    monkeypatch.setattr("sys.argv", ["scansloudness", "--window", "1.5"])
    Audiobook.main()
    for name in (NAME, STORY):
        assert np.load(Loudness.directory(data, (75, 0)) / f"{name}.npy").shape == (3, 75)
