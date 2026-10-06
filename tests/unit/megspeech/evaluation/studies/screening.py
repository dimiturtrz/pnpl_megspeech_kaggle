from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from megspeech.evaluation.studies.screening import Family, Run, Screening
from megspeech.training.config import RunConfig

RISING = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2])
EARLY = np.array([0.1, 0.5, 0.6, 0.4, 0.3, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1])


def run(name: str, val: np.ndarray, seed: int = 0, recipe: str = "r") -> Run:
    return Run(name, "fam", recipe, seed, val)


def family() -> Family:
    return Family(
        "fam", tuple(run(f"r{i}", RISING * (1 + i / 10) if i % 2 else EARLY * (1 + i / 10)) for i in range(6))
    )


def test_of(tmp_path: Path) -> None:
    assert Run.of(tmp_path) is None


def test_family_of() -> None:
    assert Run.family_of(RunConfig()) == "e40-raw-deep-carved"


def test_curves() -> None:
    assert family().curves.shape == (6, 12)


def test_best() -> None:
    assert family().best[1] == pytest.approx(1.2 * 1.1)


def test_best_epochs() -> None:
    assert family().best_epochs().tolist() == [3, 12, 3, 12, 3, 12]


def test_epoch_quantiles() -> None:
    assert family().epoch_quantiles()["q0.5"] == 7.5


def test_stops() -> None:
    stops = Family.stops(np.stack([RISING, EARLY]), 3)
    assert stops.tolist() == [11, 5]


def test_patience() -> None:
    row = family().patience(3)
    assert row["saved"] == pytest.approx(3.0)
    assert row["lost_max"] == pytest.approx(0.0)
    assert row["stopped"] == pytest.approx(0.5)


def test_leaders() -> None:
    assert Family.leaders(np.array([3.0, 2.0, 1.0]), np.array([1.0, 2.0, 3.0]), 3)
    assert not Family.leaders(np.array([3.0, 2.0, 1.0]), np.array([1.0, 2.0, 3.0]), 1)


def test_agreement() -> None:
    row = family().agreement(10, best_so_far=True)
    assert row["spearman"] == pytest.approx(1.0)
    assert row["top1"] == 1.0


def test_horizons() -> None:
    assert family().horizons(best_so_far=False).index.tolist() == [5, 10]


def test_grouped() -> None:
    runs = [run("a", RISING), run("b", RISING, recipe="s")]
    assert {key: len(group) for key, group in Screening.grouped(runs, lambda r: r.recipe).items()} == {"r": 1, "s": 1}


def test_discover(tmp_path: Path) -> None:
    assert Screening.discover(tmp_path).families == ()


def screening() -> Screening:
    runs = [run(f"r{i}", RISING * (1 + i / 10) if i % 2 else EARLY * (1 + i / 10), seed=i % 3) for i in range(6)]
    return Screening(runs, min_group=3)


def test_epochs() -> None:
    assert screening().epochs().loc["fam", "n"] == 6


def test_patiences() -> None:
    assert screening().patiences().shape[0] == 3


def test_rankings() -> None:
    assert screening().rankings(best_so_far=True).index.names == ["family", "k"]


def test_seed_groups() -> None:
    groups = screening().seed_groups()
    assert groups["n"].tolist() == [6]
    assert groups["sd"].iloc[0] > 0


def test_pooled() -> None:
    row = Screening.pooled(pd.DataFrame({"n": [3, 3], "sd": [0.1, 0.1]}))
    assert row["pooled_sd"] == pytest.approx(0.1)


def test_seed_scatter() -> None:
    assert screening().seed_scatter().index.tolist() == ["fam", "ALL"]


def test_report() -> None:
    assert "(c) pooled seed SD" in screening().report()
