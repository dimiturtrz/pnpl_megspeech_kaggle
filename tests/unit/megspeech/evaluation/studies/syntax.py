import numpy as np
import pandas as pd
import pytest

from megspeech.data.labels.phonemes import UNKNOWN
from megspeech.evaluation.studies.syntax import Probe, Probed, Table, Timing

WORDS = pd.DataFrame(
    {
        "run": ["a", "a", "a", "b"],
        "sentence": [0, 0, 0, 0],
        "onset": [1.5, 1.0, 2.5, 4.0],
        "end": [3.0, 3.0, 3.0, 5.0],
    },
    index=[7, 3, 9, 1],
)


def table(seed: int, n: int = 120) -> Table:
    """Two tags; timing separates them, the embedding is noise."""
    generator = np.random.default_rng(seed)
    tags = np.arange(n) % 2
    timing = np.c_[tags + 0.1 * generator.standard_normal(n), generator.standard_normal(n)]
    noise = generator.standard_normal((n, 4))
    return Table(
        {"timing": timing, "meg-word": noise, "meg-context": noise},
        tags,
        np.arange(n) % 3,
        np.arange(n) % 4 == 0,
        np.where(np.arange(n) < n // 2, "x", "y"),
    )


def test_features() -> None:
    features = Timing.features(WORDS)
    assert features.tolist() == [
        [1.0, 0.5, 1.0, 3.0, pytest.approx(1 / 3)],
        [0.5, 0.0, 0.0, 3.0, 0.0],
        [0.5, 1.0, 2.0, 3.0, pytest.approx(2 / 3)],
        [1.0, 0.0, 0.0, 1.0, 0.0],
    ]


def test_scores() -> None:
    probed = Probed(np.array([0, 1, 1, 2]), np.array([True, True, False, False]), np.array(["x", "x", "y", "y"]))
    scores = probed.scores(np.array([0, 1, 0, 2]))
    assert scores["all"] == (pytest.approx((1 + 0.5 + 1) / 3), 0.75, 4)
    assert scores["in-50"] == (1.0, 1.0, 2)
    assert scores["outside"] == (0.5, 0.5, 2)
    assert set(scores) == {"all", "in-50", "outside", "x", "y"}


def test_joined() -> None:
    piece = (np.zeros((3, 5)), np.ones((3, 2)), np.ones((3, 4)), np.array([2, UNKNOWN, 5]), "r", np.ones(3, bool), "d")
    joined = Table.joined([piece, piece])
    assert joined.tags.tolist() == [2, 5, 2, 5]
    assert joined.features["meg-context"].shape == (4, 4)
    assert joined.runs.tolist() == ["r"] * 4
    assert joined.domain.tolist() == ["d"] * 4


def test_model() -> None:
    assert type(Probe.model("timing")).__name__ == "HistGradientBoostingClassifier"
    assert type(Probe.model("meg-word")).__name__ == "Pipeline"


def test_probabilities() -> None:
    folded, valid = Probe(table(0), table(1)).probabilities("timing")
    assert folded.shape == valid.shape == (120, 2)
    assert np.allclose(np.exp(valid).sum(1), 1, atol=1e-4)


def test_predicted() -> None:
    probe = Probe(table(0), table(1))
    probed = {name: probe.probabilities(name) for name in ("timing", "meg-word")}
    assert (probe.predicted(("timing",), probed) == probe.valid.tags).mean() > 0.9
    assert probe.predicted(("timing", "meg-word"), probed).shape == (120,)


def test_recalls() -> None:
    probed = Probed(np.array([0, 1, 1, 2]), np.ones(4, bool), np.array(["x"] * 4))
    assert probed.recalls(np.array([0, 1, 0, 0])) == {"ADJ": (1.0, 1), "ADP": (0.5, 2), "ADV": (0.0, 1)}


def test_majority() -> None:
    assert Probed(np.array([0, 1, 1, 2]), np.ones(4, bool), np.array(["x"] * 4)).majority() == 0.5


def test_report() -> None:
    probe = Probe(table(0), table(1))
    report = probe.report()
    assert set(report) == {"timing", "meg-word", "meg-context", "timing+meg-word", "timing+meg-context"}
    assert (report["timing"] == probe.valid.tags).mean() > (report["meg-word"] == probe.valid.tags).mean()
