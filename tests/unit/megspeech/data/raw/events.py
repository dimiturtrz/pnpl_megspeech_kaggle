from pathlib import Path

import pandas as pd

from megspeech.data.raw.events import Event, Events


def table(tmp_path: Path) -> Events:
    rows = [
        ("word", "The", 0.5, "x"),
        ("phoneme", "dh_B", 0.5, "x"),
        ("word", "", 1.0, "x"),
        ("word", "null", 2.0, "NA"),
    ]
    path = tmp_path / "events.tsv"
    pd.DataFrame(rows, columns=["kind", "segment", "timemeg", "other"]).to_csv(path, sep="\t", index=False)
    return Events(path)


def test_read(tmp_path: Path) -> None:
    read = table(tmp_path).read(Event.ONSET, Event.KIND)
    assert list(read.columns) == ["kind", "segment", "timemeg"]
    assert read["segment"].isna().tolist() == [False, False, True, False]
    assert read["segment"].iloc[3] == "null"


def test_heard(tmp_path: Path) -> None:
    assert Events.heard(table(tmp_path).read()).tolist() == [True, False, False, True]


def test_words(tmp_path: Path) -> None:
    words = table(tmp_path).words(Event.ONSET)
    assert words["segment"].tolist() == ["The", "null"]
    assert words["timemeg"].tolist() == [0.5, 2.0]


def test_ordered() -> None:
    table = pd.DataFrame(
        {
            "kind": ["phoneme", "word", "phoneme", "word"],
            "segment": ["a", "A", "b", "B"],
            "timemeg": [1.0, 1.0, 1.0, 0.5],
        }
    )
    ordered = Events.ordered(table, [Event.SEGMENT, Event.ONSET])
    assert list(ordered.columns) == ["segment", "timemeg"]
    assert ordered["segment"].tolist() == ["B", "A", "a", "b"]
    assert ordered.index.tolist() == [0, 1, 2, 3]
