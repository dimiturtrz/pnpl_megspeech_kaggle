from dataclasses import replace
from pathlib import Path

import pytest
import torch

from core.paths import DataRoot
from megspeech.data.batching.split import FEATURES
from megspeech.data.config import DataConfig, Val
from megspeech.data.raw.runs import Partition
from megspeech.evaluation.reader import Reader
from megspeech.evaluation.text import RULER
from megspeech.model.input.subjects import Reading
from tests.unit.megspeech.evaluation.loaded import CPU, NAMES, POOLED, SPOKEN, TESTED, runs

PAST = torch.iinfo(torch.int64).max


def reader(tmp_path: Path, reading: Reading = Reading.OWN) -> Reader:
    return Reader(DataRoot(tmp_path), DataConfig(), CPU, reading)


def test_load(tmp_path: Path) -> None:
    source, words = reader(tmp_path).load(runs(tmp_path)[POOLED:TESTED])
    assert source.names == NAMES[POOLED:TESTED]
    assert words["position"].tolist() == [0, 1, 2, 0, 1, 2] * 2


def test_features(tmp_path: Path) -> None:
    assert reader(tmp_path).features.directory is None
    frozen = replace(reader(tmp_path), data=DataConfig(features="frozen"))
    assert frozen.features.directory == DataRoot(tmp_path).kind("models") / FEATURES / "frozen"


def test_splits(tmp_path: Path) -> None:
    every = runs(tmp_path)
    splits = reader(tmp_path, Reading.UNSEEN).splits(every[POOLED:TESTED])
    assert list(splits) == every[POOLED:TESTED]
    rows, split = splits[every[POOLED]]
    assert (rows["run"] == NAMES[POOLED]).all()
    assert split.subject.tolist() == [PAST] * len(SPOKEN)


def test_read(tmp_path: Path) -> None:
    """A run reads its MEG windows as `reading`'s subject, or fails loudly where a features model has none saved."""
    every = runs(tmp_path)
    (_, split), *_ = reader(tmp_path).splits(every[POOLED : POOLED + 1]).values()
    assert reader(tmp_path).read(NAMES[POOLED], split).subject.tolist() == [1] * len(SPOKEN)
    assert reader(tmp_path, Reading.UNSEEN).read(NAMES[POOLED], split).subject.tolist() == [PAST] * len(SPOKEN)
    with pytest.raises(FileNotFoundError):
        replace(reader(tmp_path), data=DataConfig(features="frozen")).read(NAMES[POOLED], split)


def test_domains(tmp_path: Path) -> None:
    every = runs(tmp_path)
    domains, reference = reader(tmp_path).domains({RULER: every[:1], "TheMoth": every[1:2]})
    assert set(domains) == {RULER, "TheMoth"}
    assert (reference["run"] == NAMES[0]).all()
    assert len(reference) == len(domains[RULER].y)
    unruled, empty = reader(tmp_path).domains({"TheMoth": every[1:2]})
    assert set(unruled) == {"TheMoth"}
    assert empty.empty


def test_ruled(tmp_path: Path) -> None:
    """A model scores the domains its own saved `val` and the partition say."""
    every, split = runs(tmp_path), reader(tmp_path)
    carved = replace(split, data=DataConfig(val=Val.CARVED))
    assert carved.ruled(every) == DataConfig(val=Val.CARVED).ruled(every, Partition.VALIDATION)
    official = replace(split, data=DataConfig(val=Val.OFFICIAL), partition=Partition.TEST)
    assert official.ruled(every) == DataConfig(val=Val.OFFICIAL).ruled(every, Partition.TEST)
