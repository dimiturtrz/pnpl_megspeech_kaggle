from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest
from spacy.language import Language
from spacy.tokens import Doc
from spacy.vocab import Vocab

from core.paths import DataRoot
from megspeech.data.labels.context_tags import CLASSES, OTHER, ContextTags, SpacyTagger, Tagger
from megspeech.data.labels.phonemes import UNKNOWN
from megspeech.data.labels.vocabulary import Vocabulary
from megspeech.data.raw.runs import Run
from tests.unit.megspeech.data.labels.phonemes import Root

HEADER = "kind\tsegment\tsentenceidx\twordidx\ttimemeg\n"
EVENTS = HEADER + (
    "silence\t\t\t\t\n"
    "word\tran\t0.0\t1.0\t1.5\n"
    "word\tShe\t0.0\t0.0\t1.0\n"
    "word\thome\t0.0\t2.0\t2.0\n"
    "word\tThe\t1.0\t0.0\t4.0\n"
    "word\tlone\t\t\t6.0\n"
)
LOOKUP = {"she": "PRON", "ran": "VERB", "home": "ADV", "the": "DET"}
DET, VERB, ADV, ADP, PRON = (CLASSES.index(c) for c in ("DET", "VERB", "ADV", "ADP", "PRON"))


class Lookup:
    def tag(self, sentences: list[list[str]]) -> list[list[str]]:
        return [[LOOKUP.get(word.lower(), "X") for word in sentence] for sentence in sentences]


class Short:
    def tag(self, sentences: list[list[str]]) -> list[list[str]]:
        return [sentence[1:] for sentence in sentences]


class Stub:
    vocab = Vocab()

    def pipe(self, docs: object, batch_size: int) -> list[Doc]:
        return [
            Doc(self.vocab, words=[t.text for t in doc], pos=["NOUN"] * len(doc)) for doc in cast("list[Doc]", docs)
        ]


def run(raw: Path) -> Run:
    meg = raw / "Sherlock1" / "derivatives" / "meg" / "sub-0_ses-1_task-Sherlock1_run-1_meg.h5"
    meg.parent.mkdir(parents=True, exist_ok=True)
    meg.write_bytes(b"x")
    found = Run(meg)
    found.events.parent.mkdir(parents=True, exist_ok=True)
    found.events.write_bytes(EVENTS.encode())
    return found


def crowd() -> pd.DataFrame:
    """Settled types: dominant, split, with a slip, rare classes; and one too rare to judge."""
    rows = [("the", "DET")] * 30 + [("like", "VERB")] * 15 + [("like", "ADP")] * 15
    rows += [("run", "VERB")] * 29 + [("run", "NOUN")] + [("john", "PROPN")] * 25 + [("one", "NUM")] * 25
    rows += [("rare", "VERB"), ("rare", "ADP"), ("rare", "ADP")]
    frame = pd.DataFrame(rows, columns=["word", "tag"])
    return frame.assign(run="r", onset=np.arange(len(frame), dtype=float))[["run", "onset", "word", "tag"]]


def test_tag() -> None:
    tagger = SpacyTagger(cast("Language", Stub()))
    assert tagger.tag([["a", "b"], ["c"]]) == [["NOUN", "NOUN"], ["NOUN"]]


def test_words(tmp_path: Path) -> None:
    words = ContextTags.words(run(tmp_path))
    assert words["word"].tolist() == ["She", "ran", "home", "The"]
    assert words["onset"].tolist() == [1.0, 1.5, 2.0, 4.0]
    assert words["sentence"].tolist() == [0, 0, 0, 1]
    assert set(words["run"]) == {"sub-0_ses-1_task-Sherlock1_run-1"}


def test_tagged(tmp_path: Path) -> None:
    words = ContextTags.words(run(tmp_path))
    assert ContextTags.tagged(words, cast("Tagger", Lookup()))["tag"].tolist() == ["PRON", "VERB", "ADV", "DET"]
    assert ContextTags.tagged(words, cast("Tagger", Short()))["tag"].tolist() == [""] * 4


def test_classified() -> None:
    classes = ContextTags.classified(crowd())
    kinds = crowd().assign(code=classes).groupby(["word", "tag"])["code"].agg(lambda c: sorted(set(c)))
    assert kinds["the", "DET"] == [DET]
    assert kinds["like", "VERB"] == kinds["like", "ADP"] == [UNKNOWN]
    assert kinds["run", "VERB"] == [VERB]
    assert kinds["run", "NOUN"] == [UNKNOWN]
    assert kinds["john", "PROPN"] == [OTHER]
    assert kinds["one", "NUM"] == [OTHER]
    assert kinds["rare", "VERB"] == [VERB]
    assert kinds["rare", "ADP"] == [ADP]


def test_build() -> None:
    built = ContextTags.build(crowd().iloc[::-1])
    assert built.onsets.tolist() == sorted(built.onsets.tolist())
    assert built.tags[0] == DET
    assert len(built.tags) == len(crowd())


def test_read(tmp_path: Path) -> None:
    read = ContextTags.read([run(tmp_path)], cast("Tagger", Lookup()))
    assert read.tags.tolist() == [PRON, VERB, ADV, DET]


def test_summary() -> None:
    summary = ContextTags.summary(crowd(), Vocabulary(("the", "like", "absent")))
    assert summary["tokens"].tolist() == [30, 30, 0]
    assert summary["classes"].tolist() == [{"DET": 1.0}, {"UNKNOWN": 1.0}, {}]


def test_path(tmp_path: Path) -> None:
    assert ContextTags.path(cast("DataRoot", Root(tmp_path))) == tmp_path / "processed" / "syntax" / "context_tags.npz"


def test_save(tmp_path: Path) -> None:
    assert ContextTags.build(crowd()).save(tmp_path / "nested" / "tags.npz").exists()


def test_context_tags_load(tmp_path: Path) -> None:
    original = ContextTags.build(crowd())
    loaded = ContextTags.load(original.save(tmp_path / "tags.npz"))
    assert loaded.runs.tolist() == original.runs.tolist()
    assert np.array_equal(loaded.onsets, original.onsets)
    assert np.array_equal(loaded.tags, original.tags)


def test_classes() -> None:
    table = ContextTags.build(crowd())
    found = table.classes(np.array(["r", "r", "q"]), np.array([0.0, 999.0, 0.0]))
    assert found.tolist() == [DET, UNKNOWN, UNKNOWN]


def test_trained(tmp_path: Path) -> None:
    mine = run(tmp_path)
    test = Run(tmp_path / "Sherlock1" / "derivatives" / "meg" / "sub-0_ses-12_task-Sherlock1_run-2_meg.h5")
    other = Run(tmp_path / "sub-5_ses-1_task-Sherlock1_run-1_meg.h5")
    assert ContextTags.trained([mine, test, other]) == [mine]


def test_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = Root(tmp_path)
    run(root.raw())
    monkeypatch.setattr(DataRoot, "from_config", staticmethod(lambda: root))
    monkeypatch.setattr(SpacyTagger, "load", classmethod(lambda cls: Lookup()))
    ContextTags.main()
    assert len(ContextTags.load(ContextTags.path(cast("DataRoot", root))).tags) == 4
