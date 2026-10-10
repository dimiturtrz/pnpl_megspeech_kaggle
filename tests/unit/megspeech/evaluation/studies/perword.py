from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch

from megspeech.data.batching.split import Split
from megspeech.data.labels.vocabulary import PRIMARY, VOCAB_SIZE
from megspeech.data.spans.spans import PAD
from megspeech.evaluation.studies.confusion import Scored
from megspeech.evaluation.studies.perword import (
    BUCKETS,
    Attributes,
    Confusions,
    Occurrences,
    Place,
    Position,
    Report,
    Scene,
)

PER_CLASS = 4
SPAN = 3
SECONDS = 0.5


def scored() -> Scored:
    """Each class four times: even classes peak on themselves, odd ones on their even neighbour."""
    labels = np.tile(np.arange(VOCAB_SIZE), PER_CLASS)
    probs = np.full((len(labels), VOCAB_SIZE), 0.001)
    for row, label in enumerate(labels):
        probs[row, label - label % 2] = 0.5
        probs[row, label] = max(probs[row, label], 0.0001 if label % 2 else 0.5)
    return Scored(labels, probs, np.where(np.arange(len(labels)) < len(labels) // 2, "A", "B"))


def place(n: int) -> Place:
    position = np.array([Position.FIRST, Position.MIDDLE, Position.LAST] * n, dtype=object)[:n]
    gap = np.linspace(0.1, 1.0, n)
    return Place(position, gap, gap[::-1].copy())


def occurrences() -> Occurrences:
    result = scored()
    return Occurrences(result, place(len(result.labels)))


def attributes() -> Attributes:
    pos = np.arange(VOCAB_SIZE) % 2
    return Attributes(pos, pos, pos * 2, np.linspace(0.1, 0.5, VOCAB_SIZE))


def split() -> Split:
    index = torch.tensor([[0, 1, 2], [3, 4, PAD]])
    onset = torch.tensor([[0.0, 0.25, 0.75], [0.0, 0.5, 0.0]])
    return cast(Split, SimpleNamespace(index=index, onset=onset, y=torch.zeros(5, dtype=torch.long)))


def test_place_of() -> None:
    found = Place.of(split(), torch.arange(5))
    assert found.position.tolist() == ["first", "middle", "last", "first", "last"]
    assert found.gap_next[:3].tolist() == [0.25, 0.5, pytest.approx(np.nan, nan_ok=True)]
    assert found.gap_prev[1] == 0.25
    assert np.isnan(found.gap_prev[0])


def test_concat() -> None:
    joined = place(SPAN).concat(place(SPAN))
    assert len(joined.position) == 2 * SPAN
    assert len(joined.gap_next) == 2 * SPAN


def test_durations(tmp_path: Path) -> None:
    events = tmp_path / "events.tsv"
    events.write_text(
        "kind\tsegment\tduration\nword\tThe\t0.2\nword\tthe\t0.4\nphoneme\tdh\t0.1\nword\tzzz\t0.9\n",
        encoding="utf-8",
    )
    found = Attributes.durations([events])
    assert found[PRIMARY.words.index("the")] == pytest.approx(0.3)
    assert np.isnan(found[PRIMARY.words.index("a")])


def test_function() -> None:
    found = Attributes(np.array([-1, 3, 0, 5]), np.zeros(4, int), np.zeros(4, int), np.zeros(4)).function()
    assert found.tolist() == [False, True, False, True]


def test_named() -> None:
    assert set(attributes().named()) == {"pos", "first-phoneme", "first-two-phonemes"}


def test_hits() -> None:
    assert occurrences().hits().mean() == pytest.approx(0.5)


def test_own_rate() -> None:
    rate = occurrences().own_rate()
    assert rate[0] == 1.0
    assert rate[1] == 0.0


def test_buckets() -> None:
    found = occurrences().buckets(occurrences().place.position)
    assert found["n"].sum() == VOCAB_SIZE * PER_CLASS
    assert found["hit10"].between(0, 1).all()


def test_by_position() -> None:
    assert list(occurrences().by_position().index) == ["first", "middle", "last"]


def test_by_interval() -> None:
    found = occurrences().by_interval("gap_next")
    assert len(found) == BUCKETS
    assert found["n"].sum() == VOCAB_SIZE * PER_CLASS


def test_by_word() -> None:
    table = occurrences().by_word(np.full(VOCAB_SIZE, 1 / VOCAB_SIZE), attributes())
    assert table["n"].eq(PER_CLASS).all()
    assert table["A-n"].add(table["B-n"]).eq(PER_CLASS).all()
    assert table["hit10"].iloc[0] == 1.0


def test_correlations() -> None:
    occ = occurrences()
    table = occ.by_word(np.linspace(0.1, 1.0, VOCAB_SIZE), attributes())
    assert set(occ.correlations(table)) == {"log-prior", "duration", "ioi", "n"}


def test_confusions_of() -> None:
    errors = Confusions.of(scored())
    assert errors.counts.diagonal().sum() == 0
    assert errors.counts[1, 0] == PER_CLASS


def test_top() -> None:
    found = Confusions.of(scored()).top(1)
    assert found[0][2] == PER_CLASS
    assert found[0][1] % 2 == 0


def test_kept() -> None:
    observed, chance = Confusions.of(scored()).kept(np.arange(VOCAB_SIZE) // 2)
    assert observed == 1.0
    assert chance < observed


def test_joined() -> None:
    assert len(Scene._joined([place(SPAN), place(SPAN)]).position) == 2 * SPAN


def test_run(tmp_path: Path) -> None:
    scene = Scene(occurrences(), np.full(VOCAB_SIZE, 1 / VOCAB_SIZE), 0.5)
    record = Report(scene, attributes(), tmp_path / "out.json").run()
    assert record["n"] == VOCAB_SIZE * PER_CLASS
    assert (tmp_path / "out.json").exists()
