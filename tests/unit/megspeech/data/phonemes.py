from pathlib import Path
from typing import cast

import pandas as pd
import pytest
import torch

from core.paths import DataRoot
from megspeech.data.corpus import Corpus
from megspeech.data.phonemes import PHONEMES, UNKNOWN, Pronunciation
from megspeech.data.runs import Run

HEADER = "kind\tsegment\tsentenceidx\twordidx\tphonemeidx\n"
EVENTS = HEADER + (
    "silence\t\t\t\t\n"
    "word\tThe\t0.0\t0.0\t\n"
    "phoneme\tdh_B\t0.0\t0.0\t0.0\n"
    "phoneme\tah_E\t0.0\t0.0\t1.0\n"
    "word\tnull\t0.0\t1.0\t\n"
    "phoneme\tn_B\t0.0\t1.0\t0.0\n"
    "word\tZorg\t1.0\t0.0\t\n"
    "phoneme\toov_S\t1.0\t0.0\t0.0\n"
    "word\tlone\t\t\t\n"
)
DH, N, IY = (PHONEMES.index(p) for p in ("dh", "n", "iy"))


class Root:
    def __init__(self, path: Path) -> None:
        self.base = path

    def raw(self) -> Path:
        return self.base / "raw"

    def processed(self) -> Path:
        return self.base / "processed"


def run(raw: Path) -> Run:
    made = Run(raw / "Sherlock1" / "meg" / "sub-0_ses-1_task-Sherlock1_run-1_meg.h5")
    made.events.parent.mkdir(parents=True, exist_ok=True)
    made.meg.parent.mkdir(parents=True, exist_ok=True)
    made.meg.write_bytes(b"")
    made.events.write_bytes(EVENTS.encode())
    return made


def table() -> Pronunciation:
    rows = [("the", "dh"), ("The", "dh"), ("THE", "iy"), ("a", "ah"), ("x", "oov")]
    return Pronunciation.build(pd.DataFrame(rows, columns=["word", "phoneme"]))


def test_initials(tmp_path: Path) -> None:
    initials = Pronunciation.initials(run(tmp_path).events)
    assert initials.values.tolist() == [["The", "dh"], ["null", "n"], ["Zorg", "oov"]]


def test_build() -> None:
    built = table()
    assert built.keys.tolist() == sorted(Corpus.token(pd.Series(["the", "a"])).tolist())
    assert built.phonemes(torch.from_numpy(Corpus.token(pd.Series(["the", "a"])))).tolist() == [
        DH,
        PHONEMES.index("ah"),
    ]


def test_read(tmp_path: Path) -> None:
    made = run(tmp_path)
    read = Pronunciation.read([made.events, made.events])
    assert read.phonemes(torch.from_numpy(Corpus.token(pd.Series(["the", "NULL", "zorg"])))).tolist() == [
        DH,
        N,
        UNKNOWN,
    ]


def test_path(tmp_path: Path) -> None:
    assert Pronunciation.path(cast("DataRoot", Root(tmp_path))) == tmp_path / "processed" / "phonemes" / "first.npz"


def test_save(tmp_path: Path) -> None:
    saved = table().save(tmp_path / "nested" / "first.npz")
    assert saved.exists()


def test_load(tmp_path: Path) -> None:
    original = table()
    loaded = Pronunciation.load(original.save(tmp_path / "first.npz"))
    assert torch.equal(loaded.keys, original.keys)
    assert torch.equal(loaded.first, original.first)


def test_to() -> None:
    moved = table().to(torch.device("cpu"))
    assert moved.first.device.type == "cpu"
    assert torch.equal(moved.keys, table().keys)


def test_phonemes() -> None:
    built = table()
    known = torch.from_numpy(Corpus.token(pd.Series(["the", "a", "zebra"]))).view(1, 3)
    assert built.phonemes(known).tolist() == [[DH, PHONEMES.index("ah"), UNKNOWN]]
    assert built.phonemes(torch.tensor([-100])).tolist() == [UNKNOWN]


def test_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = Root(tmp_path)
    run(root.raw())
    monkeypatch.setattr(DataRoot, "from_config", staticmethod(lambda: root))
    Pronunciation.main()
    assert len(Pronunciation.load(Pronunciation.path(cast("DataRoot", root))).keys) == 2
