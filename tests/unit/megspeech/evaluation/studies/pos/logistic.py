import numpy as np
from sklearn.linear_model import LogisticRegression

from megspeech.evaluation.studies.pos.logistic import TorchLogistic


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
