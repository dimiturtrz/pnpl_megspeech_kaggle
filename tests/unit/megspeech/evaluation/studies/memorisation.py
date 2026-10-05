import logging
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from megspeech.data.batching.split import Split
from megspeech.data.labels.vocabulary import Vocabulary
from megspeech.data.raw.runs import Partition, Run
from megspeech.data.spans.spans import Spans
from megspeech.evaluation.studies import memorisation as module
from megspeech.evaluation.studies.memorisation import POOLED, Bucket, Heard, Memorisation, NGrams, Tally
from tests.unit.megspeech.evaluation.loaded import SPOKEN, loaded, runs

CPU = torch.device("cpu")
TRAIN = pd.DataFrame({"run": "r", "sentence": [0, 0, 0, 0, 1, 1, 1], "word": list("abcdxyz")})
HEARD = pd.DataFrame({"run": "t", "sentence": [0, 0, 0, 0, 1, 1, 1], "word": list("abcdbcq")})


def tokens(letters: str) -> np.ndarray:
    return Vocabulary.token(pd.Series(list(letters)))


def split(letters: str, group: list[int]) -> Split:
    spans = Spans.by_sentence(np.array(group), np.arange(len(letters), dtype=float))
    return Split.from_arrays(np.zeros((len(letters), 4, 4), dtype=np.float32), tokens(letters), spans, CPU)


def test_windows() -> None:
    """Windows crossing a group boundary are dropped."""
    hashed, owner = NGrams.windows(tokens("abcd"), np.array([0, 0, 1, 1]), 2)
    assert owner.tolist() == [0, 1]
    assert len(hashed) == 2
    assert len(NGrams.windows(tokens("ab"), np.array([0, 0]), 3)[0]) == 0


def test_n_grams_of() -> None:
    grams = NGrams.of(tokens("abcd"), np.zeros(4, dtype=np.int64))
    assert len(grams.seen[1]) == 4
    assert len(grams.seen[4]) == 1


def test_longest() -> None:
    grams = NGrams.of(tokens("abcd"), np.zeros(4, dtype=np.int64))
    probe, group = tokens("bcdxab"), np.array([0, 0, 0, 0, 1, 1])
    assert grams.longest(probe, group, 2).tolist() == [3, 2]


def test_coverage() -> None:
    grams = NGrams.of(tokens("abcd"), np.zeros(4, dtype=np.int64))
    probe, group = tokens("abcxbc"), np.array([0, 0, 0, 0, 1, 1])
    share = grams.coverage(probe, group, 2)
    assert share[0] == pytest.approx(0.5)
    assert np.isnan(share[1])


def test_texts() -> None:
    keys, texts, flat, group = Heard.texts(TRAIN)
    assert keys["sentence"].tolist() == [0, 1]
    assert [len(text) for text in texts] == [4, 3]
    assert group.tolist() == [0, 0, 0, 0, 1, 1, 1]
    assert len(flat) == 7


def test_heard_of() -> None:
    assert len(Heard.of(TRAIN).sentences) == 2


def test_tags() -> None:
    """Sentence 0 of the test repeats training verbatim; sentence 1 (`bcq`) shares only the run `bc`."""
    group = [0, 0, 0, 0, 1, 1, 1]
    tagged = Heard.of(TRAIN).tags(split("abcdbcq", group), HEARD, HEARD)
    assert tagged["seen"].tolist() == [True] * 4 + [False] * 3
    assert tagged["longest"].tolist() == [4] * 4 + [2] * 3
    assert tagged["trigram"].tolist()[:4] == [1.0] * 4
    assert tagged["trigram"].tolist()[4:] == [0.0] * 3


def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source": ["A", "A", "B", "B"],
            "label": [0, 1, 0, 1],
            "hit": [True, True, True, False],
            "seen": [True, True, False, False],
            "longest": [5, 6, 1, 3],
            "trigram": [1.0, 1.0, 0.0, 0.5],
            "scored": [False, False, True, True],
        }
    )


def test_mask() -> None:
    data = frame()
    assert Bucket.SEEN.mask(data).tolist() == [True, True, False, False]
    assert Bucket.NOVEL.mask(data).tolist() == [False, False, True, True]
    assert Bucket.ALL.mask(data).all()
    assert Bucket.SCORED.mask(data).tolist() == [False, False, True, True]
    assert Bucket.SHORT.mask(data).tolist() == [False, False, True, False]
    assert Bucket.MIDDLE.mask(data).tolist() == [False, False, False, True]
    assert Bucket.LONG.mask(data).tolist() == [True, True, False, False]


def test_line() -> None:
    line = Tally(frame()).line(frame().iloc[2:])
    assert "n=2 classes=2 bacc=0.5000" in line
    assert "micro=0.5000" in line


def test_tally_report() -> None:
    lines = Tally(frame()).report()
    assert any(line.startswith(f"{POOLED} seen-sentence: n=2") for line in lines)
    assert any(line.startswith("B novel-sentence: n=2") for line in lines)
    assert not any(line.startswith("A novel-sentence") for line in lines)


def memorising(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Memorisation, list[Run]]:
    every = runs(tmp_path)
    monkeypatch.setattr(module.Run, "everywhere", classmethod(lambda *_: every))
    return Memorisation(replace(loaded(tmp_path), partition=Partition.TEST)), every


def test_words(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every labelled word of the test run is tagged; the fixture trains on no text, so none is heard."""
    study, every = memorising(tmp_path, monkeypatch)
    words = study.words(every, 0.0)
    assert len(words) == len(SPOKEN)
    assert set(words["source"]) == {"Sherlock1"}
    assert not words["seen"].any()
    assert (words["longest"] == 0).all()
    assert words["hit"].dtype == bool


def test_memorisation_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    study, every = memorising(tmp_path, monkeypatch)
    with caplog.at_level(logging.INFO, logger=module.__name__):
        study.report(every, 0.0, "label")
    assert any("label memorisation tau 0.0 Sherlock1 novel-sentence: n=6" in r.getMessage() for r in caplog.records)
