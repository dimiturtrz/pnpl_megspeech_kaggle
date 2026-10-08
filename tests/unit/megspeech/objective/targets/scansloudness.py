from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.io import wavfile

from core.paths import ENV_OVERRIDE, DataRoot
from megspeech.data.raw.runs import Run
from megspeech.data.raw.scans import SCANS, STIMULI
from megspeech.objective.targets.loudness import LENGTH, Loudness
from megspeech.objective.targets.scansloudness import Audiobook
from megspeech.objective.targets.speech import SAMPLING
from tests.unit.megspeech.data.raw.runs import meg
from tests.unit.megspeech.objective.targets.speech import SECONDS, loud

NAME = "sub-401_ses-1_task-SCANSab2_run-1"


def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DataRoot:
    monkeypatch.setenv(ENV_OVERRIDE, str(tmp_path))
    return DataRoot.from_config()


def served(data: DataRoot) -> Run:
    """A converted run whose words start at 0.94 s, 2 s and 3.8 s of the wav it played, silent until 1 s."""
    run = Run(meg(data.processed(SCANS.dataset), NAME))
    run.events.parent.mkdir(parents=True, exist_ok=True)
    events = pd.DataFrame(
        {"kind": ["word", "word", "word"], "segment": ["de:a", "de:b", "de:c"], "timechapter": [0.94, 2.0, 3.8]}
    )
    events.to_csv(run.events, sep="\t", index=False)
    stimuli = SCANS.home(data) / STIMULI
    stimuli.mkdir(parents=True, exist_ok=True)
    wavfile.write(
        stimuli / "task-audiobook2_run-01_stim.wav", SAMPLING, (loud(SECONDS, 1.0).audio * 1000).astype(np.int16)
    )
    return run


def test_wav(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = root(tmp_path, monkeypatch)
    assert (
        Audiobook(Run(meg(tmp_path, NAME))).wav(data) == SCANS.home(data) / STIMULI / "task-audiobook2_run-01_stim.wav"
    )


def test_targets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = root(tmp_path, monkeypatch)
    targets = Audiobook(served(data)).targets(data)
    assert targets.vectors.shape == (3, LENGTH)
    assert targets.present[0]


def test_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = root(tmp_path, monkeypatch)
    served(data)
    directory = Audiobook.write(data)
    assert directory == Loudness.directory(data)
    assert [path.stem for path in directory.glob("*.npy")] == [NAME]
    assert Audiobook.write(data) == directory


def test_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = root(tmp_path, monkeypatch)
    served(data)
    Audiobook.main()
    assert np.load(Loudness.directory(data) / f"{NAME}.npy").shape == (3, LENGTH)
