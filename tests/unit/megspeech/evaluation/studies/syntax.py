from dataclasses import replace
from typing import cast

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from megspeech.data.labels.phonemes import UNKNOWN
from megspeech.evaluation.studies.syntax import Grammar, Probe, Probed, Table, Timing, TorchLogistic

WORDS = pd.DataFrame(
    {
        "run": ["a", "a", "a", "b"],
        "sentence": [0, 0, 0, 0],
        "onset": [1.5, 1.0, 2.5, 4.0],
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
        np.arange(n) // 6,
        np.arange(n, dtype=float),
        np.where(np.arange(n) % 4 == 0, np.arange(n) % 5, -1),
    )


ALTERNATING = Grammar.of(np.array([0, 1, 0, 0, 1, 0]), np.array([0, 0, 0, 1, 1, 1]), np.arange(6.0), 2)


def test_gaps() -> None:
    gaps = Timing.gaps(WORDS)
    assert gaps.iloc[:2].tolist() == [1.0, 0.5]
    assert gaps.iloc[2:].isna().all()


def test_fitted() -> None:
    assert Timing.fitted(WORDS).final == 0.75


def test_features() -> None:
    features = Timing(final=2.0).features(WORDS)
    assert features.tolist() == [
        [1.0, 0.5, 1.0, 3.0, pytest.approx(1 / 3)],
        [0.5, 0.0, 0.0, 3.0, 0.0],
        [2.0, 1.0, 2.0, 3.0, pytest.approx(2 / 3)],
        [2.0, 0.0, 0.0, 1.0, 0.0],
    ]


def test_scores() -> None:
    probed = Probed(np.array([0, 1, 1, 2]), np.array([True, True, False, False]), np.array(["x", "x", "y", "y"]))
    scores = probed.scores(np.array([0, 1, 0, 2]))
    assert scores["all"] == (pytest.approx((1 + 0.5 + 1) / 3), 0.75, 4)
    assert scores["in-50"] == (1.0, 1.0, 2)
    assert scores["outside"] == (0.5, 0.5, 2)
    assert set(scores) == {"all", "in-50", "outside", "x", "y"}


def test_joined() -> None:
    keys, onsets = np.array(["r/0", "r/0", "r/1"]), np.arange(3.0)
    tags = np.array([2, UNKNOWN, 5])
    targets = np.array([3, -1, 4])
    piece = (
        np.zeros((3, 5)),
        np.ones((3, 2)),
        np.ones((3, 4)),
        tags,
        "r",
        np.ones(3, bool),
        "d",
        keys,
        onsets,
        targets,
    )
    joined = Table.joined([piece, piece])
    assert joined.sentences.tolist() == [0, 0, 1, 0, 0, 1]
    assert joined.onsets.tolist() == [0.0, 1.0, 2.0] * 2
    assert joined.tags.tolist() == [2, UNKNOWN, 5] * 2
    assert joined.features["meg-context"].shape == (6, 4)
    assert joined.runs.tolist() == ["r"] * 6
    assert joined.domain.tolist() == ["d"] * 6
    assert joined.targets.tolist() == [3, -1, 4] * 2


def test_rows() -> None:
    held = table(0)
    found = held.rows(np.array([0, 1, 2, 0]), np.array([0.0, 5.0, 5.0, 3.0]))
    assert found.tolist() == [0, -1, 5, 3]
    assert held.rows(np.array([1]), np.array([1.0])).tolist() == [1]
    with pytest.raises(ValueError, match="two words"):
        replace(held, onsets=np.zeros(len(held.tags))).rows(np.array([0]), np.array([0.0]))


def unseen(held: Table) -> Table:
    """`held` with every tenth word's tag unknown."""
    return replace(held, tags=np.where(np.arange(len(held.tags)) % 10 == 0, UNKNOWN, held.tags))


def test_known() -> None:
    assert Probe(unseen(table(0)), table(1)).known.sum() == 108
    assert Probe(table(0), table(1)).known.all()


def test_model() -> None:
    probe = Probe(table(0), table(1))
    assert type(probe.model("timing")).__name__ == "HistGradientBoostingClassifier"
    assert type(probe.model("meg-word")).__name__ == "TorchLogistic"
    assert cast("TorchLogistic", probe.model("meg-word")).balanced is False
    assert cast("TorchLogistic", Probe(table(0), table(1), balanced=True).model("meg-word")).balanced is True


def test_probabilities() -> None:
    folded, valid = Probe(unseen(table(0)), table(1)).probabilities("timing")
    assert folded.shape == (108, 2)
    assert valid.shape == (120, 2)
    assert np.allclose(np.exp(valid).sum(1), 1, atol=1e-4)


def test_predicted() -> None:
    probe = Probe(table(0), table(1))
    probed = {name: probe.probabilities(name) for name in ("timing", "meg-word")}
    assert (probe.predicted(("timing",), probed) == probe.valid.tags).mean() > 0.9
    assert probe.predicted(("timing", "meg-word"), probed).shape == (120,)


def test_grammar_of() -> None:
    grammar = Grammar.of(np.array([0, 1, 1]), np.array([0, 0, 1]), np.arange(3.0), 2)
    assert np.allclose(np.exp(grammar.transition), [[1 / 3, 2 / 3], [1 / 2, 1 / 2]])
    assert np.exp(ALTERNATING.start).tolist() == pytest.approx([0.75, 0.25])
    assert np.allclose(np.exp(ALTERNATING.transition), [[1 / 4, 3 / 4], [3 / 4, 1 / 4]])


def test_viterbi() -> None:
    assert ALTERNATING.viterbi(np.zeros((4, 2))).tolist() == [0, 1, 0, 1]
    assert ALTERNATING.viterbi(np.array([[0.0, 5.0], [0.0, 0.0]])).tolist() == [1, 0]


def test_decode() -> None:
    decoded = ALTERNATING.decode(np.zeros((5, 2)), np.array([1, 0, 1, 0, 1]), np.array([2.0, 1.0, 0.0, 0.0, 1.0]))
    assert decoded.tolist() == [0, 1, 0, 0, 1]


def test_stacked() -> None:
    probe = Probe(table(0), table(1))
    classes, posterior = probe.stacked(("timing",), {"timing": probe.probabilities("timing")})
    assert classes.tolist() == [0, 1]
    assert np.allclose(np.exp(posterior).sum(1), 1)


def test_parsed() -> None:
    probe = Probe(table(0), table(1))
    parsed = probe.parsed(("timing",), {"timing": probe.probabilities("timing")})
    assert (parsed == probe.valid.tags).mean() > 0.9


def test_recalls() -> None:
    probed = Probed(np.array([0, 1, 1, 2]), np.ones(4, bool), np.array(["x"] * 4))
    assert probed.recalls(np.array([0, 1, 0, 0])) == {"ADJ": (1.0, 1), "ADP": (0.5, 2), "ADV": (0.0, 1)}


def test_confusion() -> None:
    probed = Probed(np.array([0, 1, 1, 2]), np.array([True, True, True, False]), np.array(["x"] * 4))
    predicted = np.array([0, 1, 0, 0])
    assert probed.confusion(predicted, in_50=True).to_dict("index") == {
        "ADJ": {"ADJ": 1, "ADP": 0},
        "ADP": {"ADJ": 1, "ADP": 1},
    }
    outside = probed.confusion(predicted, in_50=False)
    assert outside.loc["ADV", "ADJ"] == 1
    assert outside.to_numpy().sum() == 1


def test_majority() -> None:
    assert Probed(np.array([0, 1, 1, 2]), np.ones(4, bool), np.array(["x"] * 4)).majority() == 0.5


def test_report() -> None:
    probe = Probe(table(0), table(1))
    report = probe.report()
    assert set(report) == {
        "timing",
        "meg-word",
        "meg-context",
        "timing+meg-word",
        "timing+meg-context",
        "timing+meg-context+grammar",
    }
    assert (report["timing"] == probe.valid.tags).mean() > (report["meg-word"] == probe.valid.tags).mean()


def overlapping(n: int = 600) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(0)
    tags = np.array([5, 7, 9])[np.arange(n) % 3]
    centres = generator.standard_normal((3, 4)) * 1.5
    return centres[np.arange(n) % 3] + generator.standard_normal((n, 4)), tags


def test_fit() -> None:
    x, y = overlapping()
    fitted = TorchLogistic(c=0.1).fit(x, y)
    assert fitted.classes_.tolist() == [5, 7, 9]
    assert fitted.weight_.shape == (4, 3)
    reference = LogisticRegression(C=0.1, max_iter=1000).fit((x - x.mean(0)) / x.std(0), y)
    assert np.allclose(fitted.weight_.cpu().numpy().T, reference.coef_, atol=2e-2)


def test_fit_balanced() -> None:
    x = np.r_[np.zeros((90, 1)), np.ones((10, 1))] + np.random.default_rng(0).standard_normal((100, 1)) * 0.8
    y = np.r_[np.zeros(90, int), np.ones(10, int)]
    natural, balanced = TorchLogistic().fit(x, y).predict(x), TorchLogistic(balanced=True).fit(x, y).predict(x)
    assert balanced.sum() > natural.sum()


def test_predict_log_proba() -> None:
    x, y = overlapping()
    reference = LogisticRegression(C=0.1, max_iter=1000).fit((x - x.mean(0)) / x.std(0), y)
    log_proba = TorchLogistic(c=0.1).fit(x, y).predict_log_proba(x)
    assert np.abs(log_proba - reference.predict_log_proba((x - x.mean(0)) / x.std(0))).max() < 2e-2
    assert np.allclose(np.exp(log_proba).sum(1), 1, atol=1e-5)


def test_predict_proba() -> None:
    x, y = overlapping()
    fitted = TorchLogistic().fit(x, y)
    assert np.allclose(fitted.predict_proba(x), np.exp(fitted.predict_log_proba(x)))


def test_predict() -> None:
    x, y = overlapping()
    reference = LogisticRegression(C=0.1, max_iter=1000).fit((x - x.mean(0)) / x.std(0), y)
    predicted = TorchLogistic(c=0.1).fit(x, y).predict(x)
    assert set(predicted) <= {5, 7, 9}
    assert (predicted == reference.predict((x - x.mean(0)) / x.std(0))).mean() > 0.99
