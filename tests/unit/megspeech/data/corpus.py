from pathlib import Path
from typing import cast

import numpy as np
import torch

from core.paths import DataRoot
from megspeech.data.corpus import Corpus, DataConfig
from megspeech.data.epochs import Epochs
from megspeech.data.runs import CHANNELS, Run

SAMPLES = 50
CPU = torch.device("cpu")


class Root:
    def __init__(self, raw: Path) -> None:
        self.path = raw

    def raw(self) -> Path:
        return self.path

    def processed(self) -> Path:
        return self.path / "processed"


def corpus_root(tmp_path: Path) -> DataRoot:
    epochs = Epochs(np.zeros((2, CHANNELS, SAMPLES), np.float32), np.array([0, 1]), np.array([0.0, 0.1]))
    names = ["sub-0_ses-1_task-Sherlock1_run-1_meg.h5", "sub-0_ses-11_task-Sherlock1_run-2_meg.h5"]
    for run in (Run(tmp_path / name) for name in names):
        run.meg.touch()
        run.events.parent.mkdir(parents=True, exist_ok=True)
        run.events.write_text("kind")
        epochs.save(tmp_path / "processed" / "epochs" / f"{run.name}.npz")
    return cast("DataRoot", Root(tmp_path))


def test_trains_on(tmp_path: Path) -> None:
    sherlock = Run(tmp_path / "sub-0_ses-1_task-Sherlock1_run-1_meg.h5")
    timit = Run(tmp_path / "sub-0_ses-1_task-TIMIT_run-1_meg.h5")
    held = Run(tmp_path / "sub-0_ses-11_task-Sherlock1_run-2_meg.h5")
    only = DataConfig(tasks=("Sherlock1",))
    assert [only.trains_on(r) for r in (sherlock, timit, held)] == [True, False, False]
    assert DataConfig().trains_on(timit)


def test_load(tmp_path: Path) -> None:
    corpus = Corpus.load(DataConfig(), corpus_root(tmp_path), CPU)
    assert len(corpus.train.y) == 2
    assert set(corpus.validation) == {"Sherlock1"}
    assert corpus.seen == {(0, 1)}
