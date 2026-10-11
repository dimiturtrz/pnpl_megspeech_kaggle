from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest
import torch
from torch import Tensor

from megspeech.data.labels.vocabulary import PRIMARY, VOCAB_SIZE
from megspeech.evaluation.studies.cause import (
    ARTICLES,
    POOLED,
    Aligned,
    Described,
    Regression,
    Stress,
    Study,
    Surprisal,
)
from megspeech.evaluation.studies.confusion import Scored
from megspeech.evaluation.studies.perword import Attributes, Occurrences, Place, Position, Scene

LM_VOCAB = 8
WINDOW = 4
PER_CLASS = 6
SPAN = 3
DRAWS = 5
ONSET = 0.3
GAP = 0.5

EVENTS = (
    "kind\tsegment\tsentenceidx\twordidx\ttimemeg\tduration\n"
    "silence\t\t\t\t0.0\t0.3\n"
    "word\tThe\t0\t0\t0.3\t0.1\n"
    "phoneme\tdh_B\t0\t0\t0.3\t0.05\n"
    "phoneme\tiy_E\t0\t0\t0.35\t0.05\n"
    "word\tapple\t0\t1\t0.4\t0.3\n"
    "phoneme\tae_B\t0\t1\t0.4\t0.1\n"
    "phoneme\tl_E\t0\t1\t0.5\t0.2\n"
    "word\ta\t0\t2\t0.8\t0.05\n"
    "phoneme\tah_S\t0\t2\t0.8\t0.05\n"
)


def uniform(ids: Tensor) -> Tensor:
    return torch.zeros(len(ids), LM_VOCAB)


def letters(texts: list[str]) -> list[list[int]]:
    return [[1] * len(text.strip()) for text in texts]


def surprisal() -> Surprisal:
    return Surprisal(uniform, letters, 0, WINDOW)


def stress() -> Stress:
    return Stress(
        {word.replace("’", "'"): [["AH0"], ["AH1"]] for word in PRIMARY.words}
        | {
            "the": [["DH", "AH0"], ["DH", "AH1"], ["DH", "IY0"]],
            "a": [["AH0"], ["EY1"]],
            "apple": [["AE1", "P", "AH0", "L"]],
        }
    )


def aligned(tmp_path: Path) -> Aligned:
    path = tmp_path / "run_events.tsv"
    path.write_text(EVENTS, encoding="utf-8")
    return Aligned.read(path)


def scored(seed: int) -> Scored:
    labels = np.tile(np.arange(VOCAB_SIZE), PER_CLASS)
    probs = np.random.default_rng(seed).normal(size=(len(labels), VOCAB_SIZE))
    probs[np.arange(len(labels)), labels] += np.where(np.arange(len(labels)) // VOCAB_SIZE % 2, 6, -2)
    return Scored(labels, probs, np.full(len(labels), "A"))


def scene() -> Scene:
    context = scored(0)
    n = len(context.labels)
    position = np.array([Position.FIRST, Position.MIDDLE, Position.LAST] * n, dtype=object)[:n]
    place = Place(position, np.full(n, GAP), np.full(n, GAP), np.arange(n) // SPAN)
    onset = ONSET + GAP * np.arange(n)
    words = pd.Series([PRIMARY.words[label] for label in context.labels])
    spoken = pd.DataFrame({"run": "r", "onset": onset, "word": words})
    prior = np.linspace(1, 2, VOCAB_SIZE) / np.linspace(1, 2, VOCAB_SIZE).sum()
    return Scene(Occurrences(context, place), prior, 0.5, scored(1), spoken)


def matching(found: Scene) -> Aligned:
    spoken = found.spoken
    assert spoken is not None
    rng = np.random.default_rng(2)
    n = len(spoken)
    return Aligned(
        pd.DataFrame(
            {
                "word": spoken["word"],
                "onset": spoken["onset"],
                "duration": rng.uniform(0.05, 0.4, n),
                "phones": [("dh", "ah") if word == "the" else ("ah",) for word in spoken["word"]],
                "next_vowel": rng.random(n) < GAP,
                "surprisal": rng.uniform(1, 15, n),
            }
        )
    )


def attributes() -> Attributes:
    pos = np.arange(VOCAB_SIZE) % 2
    return Attributes(pos, pos, pos, np.full(VOCAB_SIZE, 0.2))


def described() -> Described:
    found = scene()
    return Described.of(found, {"r": matching(found)}, stress(), attributes())


def test_bits() -> None:
    found = surprisal().bits(["the", "apple", "a", "dog", "is"])
    assert found.tolist() == pytest.approx([n * np.log2(LM_VOCAB) for n in (3, 5, 1, 3, 2)])


def test_stress_of() -> None:
    marks = stress()
    assert marks.of("the", ("dh", "ah"), function=True) == 0
    assert marks.of("the", ("dh", "ah"), function=False) == 2
    assert marks.of("a", ("ey",), function=True) == 2
    assert marks.of("a", ("zz",), function=True) == 0
    assert np.isnan(marks.of("zzz", (), function=True))


def test_read(tmp_path: Path) -> None:
    words = aligned(tmp_path).words
    assert words["word"].tolist() == ["The", "apple", "a"]
    assert words["phones"].iloc[0] == ("dh", "iy")
    assert words["next_vowel"].tolist() == [True, True, False]


def test_surprised(tmp_path: Path) -> None:
    found = aligned(tmp_path).surprised(surprisal()).words["surprisal"]
    assert found.tolist() == pytest.approx([3 * np.log2(LM_VOCAB), 5 * np.log2(LM_VOCAB), np.log2(LM_VOCAB)])


def test_at(tmp_path: Path) -> None:
    found = aligned(tmp_path).at(np.array([0.401, 0.8, 0.3]), pd.Series(["apple", "the", "the"]))
    assert found["word"].iloc[0] == "apple"
    assert pd.isna(found["word"].iloc[1])
    assert found["duration"].iloc[2] == pytest.approx(0.1)


def test_described_of() -> None:
    table = described().table
    assert len(table) == VOCAB_SIZE * PER_CLASS
    assert table[list(POOLED)].notna().all().all()
    assert set(table["stressed"].unique()) <= {0.0, 1.0}
    assert table["top10-context"].between(0, 1).all()


def test_margin() -> None:
    result = scored(0)
    margin = Described.margin(result)
    assert margin.shape == (len(result.labels),)
    assert np.abs(margin).max() <= 1


def test_articles() -> None:
    frame = described().articles("context")
    assert set(frame.index.get_level_values("word")) == set(ARTICLES)
    assert frame.loc[("the", "all", "all"), "n"] == PER_CLASS
    assert {"surprisal", "log_duration", "schwa", "next_vowel"} <= set(frame.index.get_level_values("by"))


def test_subset() -> None:
    assert len(described().subset(ARTICLES).table) == len(ARTICLES) * PER_CLASS


def test_design() -> None:
    table = described().table
    assert Regression(POOLED, fixed=False).design(table).shape == (len(table), len(POOLED))
    assert Regression(POOLED, fixed=True).design(table).shape == (len(table), len(POOLED) + VOCAB_SIZE)


def test_held() -> None:
    rows = Regression(("surprisal", "first"), fixed=False).held(described().table, "top10-context")
    assert rows["surprisal"].mean() == pytest.approx(0, abs=1e-9)
    assert set(rows["first"].unique()) == {0.0, 1.0}


def test_coefficients() -> None:
    table = described().table
    spec = Regression(("surprisal",), fixed=False)
    rows = spec.held(table.assign(signal=(table["surprisal"] > table["surprisal"].median()).astype(float)), "signal")
    assert spec.coefficients(rows, "signal")[0] > 0


def test_fit() -> None:
    frame = Regression(POOLED, fixed=True).fit(described().table, "top10-context", DRAWS, seed=0)
    assert list(frame.index) == list(POOLED)
    assert (frame["lo"] <= frame["hi"]).all()
    assert frame["n"].iloc[0] == VOCAB_SIZE * PER_CLASS


def test_clear() -> None:
    frame = pd.DataFrame({"lo": [0.1, -0.2, -0.5], "hi": [0.3, 0.2, -0.1]})
    assert Regression.clear(frame).tolist() == [True, False, True]


def test_run(tmp_path: Path) -> None:
    record = Study(described(), tmp_path / "cause.json", DRAWS).run()
    assert record["n"] == VOCAB_SIZE * PER_CLASS
    assert "context/top10/word-fixed" in cast(dict[str, object], record["fits"])
    assert (tmp_path / "cause.json").exists()
