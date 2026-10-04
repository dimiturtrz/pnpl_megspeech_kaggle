from pathlib import Path
from typing import cast

import pandas as pd
import pytest
import torch

from core.paths import DataRoot
from megspeech.data.labels.phonemes import UNKNOWN
from megspeech.data.labels.syntax import TAGS, PartsOfSpeech
from megspeech.data.labels.vocabulary import Vocabulary
from tests.unit.megspeech.data.labels.phonemes import Root

HEADER = "kind\tsegment\tsentenceidx\twordidx\n"
EVENTS = HEADER + (
    "silence\t\t\t\n"
    "word\tran\t0.0\t1.0\n"
    "word\tShe\t0.0\t0.0\n"
    "word\thome\t0.0\t2.0\n"
    "word\tThe\t1.0\t0.0\n"
    "word\tdog\t1.0\t1.0\n"
    "word\tbarked\t1.0\t2.0\n"
    "word\tlone\t\t\n"
)
NOUN, VERB, DET, PRON = (TAGS.index(t) for t in ("NOUN", "VERB", "DET", "PRON"))


def events(raw: Path) -> Path:
    path = raw / "Sherlock1" / "derivatives" / "events" / "sub-0_ses-1_task-Sherlock1_run-1_events.tsv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(EVENTS.encode())
    return path


def tokens(*words: str) -> torch.Tensor:
    return torch.from_numpy(Vocabulary.token(pd.Series(words)))


def table() -> PartsOfSpeech:
    rows = [("like", "VERB"), ("Like", "ADP"), ("LIKE", "VERB"), ("the", "DET"), ("x", "NOPE")]
    return PartsOfSpeech.build(pd.DataFrame(rows, columns=["word", "tag"]))


def test_tagger() -> None:
    assert "tagger" in PartsOfSpeech.tagger().pipe_names


def test_sentences(tmp_path: Path) -> None:
    assert PartsOfSpeech.sentences(events(tmp_path)) == [["She", "ran", "home"], ["The", "dog", "barked"]]


def test_tagged() -> None:
    tagged = PartsOfSpeech.tagged([["The", "dog", "barked"]], PartsOfSpeech.tagger())
    assert tagged.values.tolist() == [["The", "DET"], ["dog", "NOUN"], ["barked", "VERB"]]


def test_build() -> None:
    built = table()
    assert built.keys.tolist() == sorted(tokens("like", "the").tolist())
    assert built.classes(tokens("like", "the")).tolist() == [VERB, DET]


def test_read(tmp_path: Path) -> None:
    path = events(tmp_path)
    read = PartsOfSpeech.read([path, path], PartsOfSpeech.tagger())
    assert read.classes(tokens("she", "DOG", "barked", "lone")).tolist() == [PRON, NOUN, VERB, UNKNOWN]


def test_path(tmp_path: Path) -> None:
    assert PartsOfSpeech.path(cast("DataRoot", Root(tmp_path))) == tmp_path / "processed" / "syntax" / "tags.npz"


def test_save(tmp_path: Path) -> None:
    assert table().save(tmp_path / "nested" / "tags.npz").exists()


def test_load(tmp_path: Path) -> None:
    original = table()
    loaded = PartsOfSpeech.load(original.save(tmp_path / "tags.npz"))
    assert torch.equal(loaded.keys, original.keys)
    assert torch.equal(loaded.tags, original.tags)


def test_to() -> None:
    moved = table().to(torch.device("cpu"))
    assert moved.tags.device.type == "cpu"
    assert torch.equal(moved.keys, table().keys)


def test_classes() -> None:
    known = tokens("the", "zebra").view(1, 2)
    assert table().classes(known).tolist() == [[DET, UNKNOWN]]
    assert table().classes(torch.tensor([-100])).tolist() == [UNKNOWN]


def test_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = Root(tmp_path)
    events(root.raw())
    monkeypatch.setattr(DataRoot, "from_config", staticmethod(lambda: root))
    PartsOfSpeech.main()
    assert len(PartsOfSpeech.load(PartsOfSpeech.path(cast("DataRoot", root))).keys) == 6
