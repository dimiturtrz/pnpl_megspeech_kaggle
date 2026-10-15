import numpy as np
import pandas as pd
import pytest

from megspeech.data.labels.vocabulary import VOCAB_SIZE
from megspeech.data.raw.runs import NO_SENTENCE
from megspeech.data.stratify import IOIS, KNOWN, LOG_IOI, SENTENCED, SENTENCES, WORDS, Stratify
from megspeech.evaluation.viz.sample import LENGTHS, Picker, Profiles, Sentences, Strata

MOCHA = "sub-0_ses-2_task-MOCHATIMIT_run-1"
SHERLOCK = "sub-0_ses-11_task-Sherlock1_run-2"
BOOKS = ("MOCHATIMIT", "Sherlock1")


def words() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "run": [MOCHA, MOCHA, MOCHA, SHERLOCK, SHERLOCK],
            "sentence": [0, 0, 0, NO_SENTENCE, 1],
            "onset": [1.0, 1.5, 2.5, 4.0, 6.0],
            "word": ["the", "cat", "is", "a", "it"],
            "target": [1, 2, 0, 2, 4],
        }
    )


def test_profiles_of() -> None:
    profiles = Profiles.of(words(), BOOKS)
    assert profiles.shape == (5, KNOWN + 1 + VOCAB_SIZE + len(BOOKS))
    assert profiles[:, WORDS].tolist() == [1, 1, 1, 1, 1]
    assert profiles[:, SENTENCES].sum() == pytest.approx(2)
    assert profiles[:, SENTENCED].tolist() == [1, 1, 1, 0, 1]
    assert profiles[:, IOIS].tolist() == [0, 1, 1, 0, 0]
    assert profiles[:, LOG_IOI] == pytest.approx([0, np.log(0.5), 0, 0, 0])
    assert profiles[:, KNOWN + 1 + 2].tolist() == [0, 1, 0, 1, 0]
    assert profiles[:, -2:].tolist() == [[1, 0], [1, 0], [1, 0], [0, 1], [0, 1]]


def test_quotas() -> None:
    sources = pd.Series(["a"] * 5 + ["b"] * 3 + ["c"] * 2)
    assert Picker.quotas(sources, 5) == {"a": 3, "b": 1, "c": 1}
    assert Picker.quotas(sources, 100) == {"a": 5, "b": 3, "c": 2}


def profiles(rows: int = 60) -> np.ndarray:
    rng = np.random.default_rng(1)
    frame = pd.DataFrame(
        {
            "run": [MOCHA] * rows,
            "sentence": np.repeat(np.arange(rows // 6), 6),
            "onset": np.cumsum(rng.uniform(0.1, 0.5, rows)),
            "target": rng.integers(0, 5, rows),
        }
    )
    return Profiles.of(frame, BOOKS)


def test_cost() -> None:
    table = profiles()
    total = table.sum(0)
    held = np.stack([table[:30].sum(0), table[:2].sum(0)])
    costs = Picker().cost(held, total, 0.5, np.ones(Stratify.gaps(table[:1], table[1:2], 0.5).shape[1]))
    assert costs.shape == (2,)
    assert np.all(costs >= 0)


def test_chosen() -> None:
    table = profiles()
    picked = Picker(draws=32, rounds=4, proposals=64).chosen(table, 12)
    assert len(picked) == len(set(picked.tolist())) == 12
    assert Picker().chosen(table[:5], 9).tolist() == [0, 1, 2, 3, 4]


def test_swapped() -> None:
    table = profiles()
    mask = np.zeros(len(table), dtype=bool)
    mask[:12] = True
    picker = Picker(proposals=256)
    scale = np.ones(Stratify.gaps(table[:1], table[1:2], 0.5).shape[1])
    total = table.sum(0)
    before = picker.cost(table[mask].sum(0)[None], total, 0.2, scale)[0]
    swapped = picker.swapped(table, mask.copy(), scale, np.random.default_rng(0))
    assert swapped.sum() == 12
    assert picker.cost(table[swapped].sum(0)[None], total, 0.2, scale)[0] <= before


def test_pick() -> None:
    table = profiles()
    sources = pd.Series(["a"] * 40 + ["b"] * 20)
    picked = Picker(draws=16, rounds=2, proposals=32).pick(table, sources, 9)
    assert picked.tolist() == sorted(picked.tolist())
    assert (picked < 40).sum() == 6
    assert (picked >= 40).sum() == 3


def test_whole() -> None:
    table = profiles()
    keys = pd.Series([f"s{i // 3}" for i in range(60)])
    sources = pd.Series(["a"] * 30 + ["b"] * 30)
    picked = Picker(draws=16, rounds=2, proposals=32).whole(table, keys, sources, 4)
    assert len(picked) == 12
    assert set(keys[picked].value_counts()) == {3}
    assert (picked < 30).sum() == 6


def test_keys() -> None:
    assert Sentences.keys(words()).tolist() == [f"{MOCHA}/0"] * 3 + [f"{SHERLOCK}@4.0", f"{SHERLOCK}/1"]


def test_lengths() -> None:
    table = words()
    assert Sentences(table).lengths(table).tolist() == [3, 3, 3, 1, 1]


def test_text() -> None:
    table = words()
    sentences = Sentences(table)
    assert sentences.text(table.iloc[1]) == ("the cat is", (4, 7))
    assert sentences.text(table.iloc[3]) == ("a", (0, 1))


def test_strata_of() -> None:
    table = words()
    strata = Strata.of(table, np.array([3, 3, 3, 1, 1]), np.array([1, 2, 3, 4, 5, 6]))
    assert [s["source"] for s in strata] == ["MOCHATIMIT"] * 3 + ["Sherlock"] * 2
    assert [s["length"] for s in strata] == [LENGTHS[1]] * 3 + [LENGTHS[0]] * 2
