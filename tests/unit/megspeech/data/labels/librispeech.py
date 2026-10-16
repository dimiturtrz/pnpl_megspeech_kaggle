import json
import logging
import sqlite3
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from core.paths import DataRoot
from megspeech.data.labels import librispeech as module
from megspeech.data.labels.librispeech import (
    Calibration,
    Chapters,
    Corpus,
    Cutter,
    Punctuated,
    Report,
    Rows,
    Tally,
    Utterance,
    Word,
    main,
)
from megspeech.data.raw.librispeech_alignments import Fetch

CHAPTERS = """; header
;
100 | 7 | 5.0 | train-clean-100 | 11 | 1001 | One | Study
200 | 8 | 5.0 | train-clean-100 | 22 | 1002 | Two | Ghost
300 | 9 | 5.0 | dev-clean | 33 | 1003 | Three | Other
"""
ALIGNED = [
    {"start": 0.0, "end": 0.4, "word": "hello"},
    {"start": 0.5, "end": 0.9, "word": "<unk>"},
    {"start": 2.0, "end": 2.4, "word": "there"},
    {"start": 2.5, "end": 2.9, "word": "friend"},
]


def utterance(uid: str = "7-100-0001", starts: tuple[float, ...] = (0.0, 0.5, 2.0, 2.5)) -> Utterance:
    return Utterance(uid, tuple(Word(s, s + 0.4, "w") for s in starts))


def annotated(path: Path) -> Path:
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE authors (id INTEGER, first_name TEXT, last_name TEXT)")
        db.execute("CREATE TABLE project_author_associations (author_id INTEGER, project_id INTEGER)")
        db.executemany("INSERT INTO authors VALUES (?, ?, ?)", [(1, "Sir Arthur Conan", "Doyle"), (2, "John", "Doyle")])
        db.executemany("INSERT INTO project_author_associations VALUES (?, ?)", [(1, 11), (2, 22)])
    return path


def world(tmp_path: Path) -> DataRoot:
    root = DataRoot(tmp_path)
    fetch = Fetch(root)
    fetch.meta.mkdir(parents=True)
    (fetch.meta / "CHAPTERS.TXT").write_text(CHAPTERS, encoding="utf-8")
    annotated(fetch.meta / "lv-annotated.sqlite.db")
    rows = [
        {"audio_filepath": "dev-clean/7/300/7-300-0000.flac", "text": "Hello, world. Mr. Smith left? Yes!"},
        {"audio_filepath": "dev-clean/7/300/7-300-0001.flac", "text": "Something else entirely."},
    ]
    (fetch.meta / "dev-clean.json").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    words = lambda *ws: [{"start": 0.2 * i, "end": 0.2 * i + 0.1, "word": w} for i, w in enumerate(ws)]  # noqa: E731
    shard = pa.table(
        {
            "id": ["7-300-0000", "7-300-0001", "7-300-0002", "7-100-0000"],
            "transcript": ["hello world mr smith left yes", "other words here", "no punctuation here", "doyle text"],
            "words": [
                words("hello", "world", "mr", "smith", "left", "yes"),
                words("other", "words", "here"),
                [*words("no", "punctuation"), {"start": 5.0, "end": 5.1, "word": "here"}],
                words("doyle", "text"),
            ],
        }
    )
    fetch.home.mkdir(parents=True, exist_ok=True)
    pq.write_table(shard, fetch.home / "dev_clean-00000-of-00001.parquet")
    return root


def test_from_row() -> None:
    restored = Utterance.from_row("7-100-0001", "Hello Mystery there friend", ALIGNED)
    assert [w.text for w in restored.words] == ["hello", "mystery", "there", "friend"]
    kept = Utterance.from_row("7-100-0001", "too short", ALIGNED)
    assert [w.text for w in kept.words][1] == "<unk>"
    assert restored.words[2] == Word(2.0, 2.4, "there")


def test_speaker() -> None:
    assert utterance().speaker == 7


def test_chapter() -> None:
    assert utterance().chapter == 100


def test_gaps() -> None:
    assert utterance().gaps() == pytest.approx([0.1, 1.1, 0.1])
    assert utterance(starts=(0.0,)).gaps() == []


def test_parse() -> None:
    parsed = Punctuated.parse('"Well-known," said Mr. Smith, J. M. Barrie’s friend... Really? I do.')
    assert parsed.words == ("well", "known", "said", "mr", "smith", "j", "m", "barrie's", "friend", "really", "i", "do")
    assert [w for w, end in zip(parsed.words, parsed.ends, strict=True) if end] == ["friend", "really", "do"]
    assert Punctuated.parse("-- ' . yes").words == ("yes",)


@pytest.mark.parametrize(
    ("raw", "last", "expected"),
    [
        ("done.", "done", True),
        ("done?”", "done", True),
        ("done!", "done", True),
        ("done,", "done", False),
        ("Mr.", "mr", False),
        ("J.", "j", False),
        ("I.", "i", True),
        ("him.)", "him", True),
    ],
)
def test_closes(raw: str, last: str, expected: bool) -> None:
    assert Punctuated.closes(raw, last) is expected


def test_matches() -> None:
    heard = Utterance("7-100-0001", (Word(0, 1, "grocers"), Word(1, 2, "<unk>"), Word(2, 3, "'em")))
    assert Punctuated(("grocers", "anything", "em"), (False,) * 3).matches(heard)
    assert not Punctuated(("grocers", "anything"), (False,) * 2).matches(heard)
    assert not Punctuated(("grocer", "anything", "em"), (False,) * 3).matches(heard)


def test_cut() -> None:
    text = Punctuated(("a", "b", "c", "d"), (False, True, False, True))
    matching = Utterance("7-1-0", tuple(Word(i, i + 0.5, w) for i, w in enumerate("abcd")))
    assert Cutter(0.3).cut(matching, text) == ([range(0, 2), range(2, 4)], True)
    paused = utterance()
    assert Cutter(0.5).cut(paused, None) == ([range(0, 2), range(2, 4)], False)
    assert Cutter(2.0).cut(paused, text) == ([range(0, 4)], False)
    assert Cutter(2.0).cut(paused, None) == ([range(0, 4)], False)


def test_ranges() -> None:
    assert Cutter.ranges([2, 4, 4], 4) == [range(0, 2), range(2, 4)]
    assert Cutter.ranges([], 3) == [range(0, 3)]


def test_read(tmp_path: Path) -> None:
    (tmp_path / "CHAPTERS.TXT").write_text(CHAPTERS, encoding="utf-8")
    chapters = Chapters.read(tmp_path / "CHAPTERS.TXT", annotated(tmp_path / "lv.db"))
    assert chapters.book == {100: 1001, 200: 1002, 300: 1003}
    assert chapters.barred == frozenset({100})


def test_excluded() -> None:
    chapters = Chapters({1: 5, 2: 6}, frozenset({1}))
    assert chapters.excluded(1)
    assert not chapters.excluded(2)


def test_tally_add() -> None:
    tally = Tally()
    tally.add("kept", 3)
    tally.add("kept", 2)
    assert (tally.counts["kept_utterances"], tally.counts["kept_words"]) == (2, 5)


def test_rows_add() -> None:
    rows = Rows()
    rows.add(
        3, Utterance("7-100-0001", (Word(1.0, 1.5, "a"), Word(2.0, 2.5, "b"), Word(3.0, 3.5, "c"))), range(1, 3), 9
    )
    assert rows.columns["onset"] == [0.0, 1.0]
    assert rows.columns["span"] == [3, 3]
    assert rows.columns["word"] == ["b", "c"]


def test_tally_share() -> None:
    tally = Tally()
    tally.add("a", 3)
    tally.add("b", 1)
    assert tally.share("a", ("a", "b")) == 0.75
    assert Tally().share("a", ("a",)) == 0.0


def test_report_share() -> None:
    table = pa.table({"word": ["<unk>", "a", "b", "c"]})
    assert Report.share(table, "<unk>") == 0.25


def test_utterances(tmp_path: Path) -> None:
    corpus = Corpus(world(tmp_path))
    found = corpus.utterances(corpus.fetch.home / "dev_clean-00000-of-00001.parquet")
    assert [u.uid for u in found][:2] == ["7-300-0000", "7-300-0001"]
    assert found[0].words[2].text == "mr"


def test_texts(tmp_path: Path) -> None:
    corpus = Corpus(world(tmp_path))
    texts = corpus.texts(corpus.fetch.home / "dev_clean-00000-of-00001.parquet")
    assert set(texts) == {"7-300-0000", "7-300-0001"}
    assert texts["7-300-0000"].words[:3] == ("hello", "world", "mr")


def test_threshold(tmp_path: Path) -> None:
    corpus = Corpus(world(tmp_path))
    assert corpus.threshold([corpus.fetch.home / "dev_clean-00000-of-00001.parquet"]) in module.PAUSE_GRID


def test_build(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    out = Corpus(world(tmp_path), pause=0.5).build()
    table = pq.read_table(out).to_pandas()
    assert table["utterance"].unique().tolist() == ["7-300-0000", "7-300-0001", "7-300-0002"]
    first = table[table["utterance"] == "7-300-0000"]
    assert first["word"].tolist() == ["hello", "world", "mr", "smith", "left"]
    assert first["span"].nunique() == 2
    assert first["onset"].iloc[0] == 0.0
    assert first.groupby("span")["onset"].min().tolist() == [0.0, 0.0]
    assert set(table["book"]) == {1003}
    assert "excluded" in caplog.text


def test_rows(tmp_path: Path) -> None:
    corpus = Corpus(world(tmp_path))
    meta = corpus.fetch.meta
    chapters = Chapters.read(meta / "CHAPTERS.TXT", meta / "lv-annotated.sqlite.db")
    tally = Tally()
    shard = corpus.fetch.home / "dev_clean-00000-of-00001.parquet"
    rows, span = corpus.rows(shard, Cutter(0.5), chapters, tally, 5)
    assert rows.columns["span"][0] == 5
    assert rows.columns["span"][-1] == span - 1
    assert tally.counts["kept_words"] == len(rows.columns["word"])


def test_table() -> None:
    rows = Rows()
    rows.add(0, utterance(), range(0, 2), 5)
    table = rows.table()
    assert table.schema == module.SCHEMA
    assert table.num_rows == 2


def test_best() -> None:
    gaps = np.array([0.05, 0.1, 0.12, 0.8, 0.9, 0.15])
    ends = np.array([False, False, False, True, True, False])
    assert 0.15 <= Calibration(gaps, ends).best() < 0.8


def test_f1() -> None:
    calibration = Calibration(np.zeros(4), np.array([True, True, False, False]))
    assert calibration.f1(np.array([True, False, False, False])) == pytest.approx(2 / 3)
    assert calibration.f1(np.zeros(4, dtype=bool)) == 0.0


def test_log(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    corpus = Corpus(world(tmp_path), pause=0.5)
    out = corpus.build()
    caplog.clear()
    tally = Tally()
    tally.add("matched", 4)
    Report(out, tally, Chapters({1: 2}, frozenset())).log()
    assert "words/span median" in caplog.text
    assert "PRIMARY vocabulary" in caplog.text


def test_vocabulary() -> None:
    counted = Report.vocabulary(pa.table({"word": ["the", "The", "the", "zzz", "a"]}))
    assert counted["the"] == 3
    assert counted["a"] == 1
    assert counted["is"] == 0


def test_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = world(tmp_path)
    monkeypatch.setattr(module.DataRoot, "from_config", classmethod(lambda cls: root))
    main(["--pause", "0.5"])
    assert (root.processed(module.DATASET) / module.OUTPUT).exists()
