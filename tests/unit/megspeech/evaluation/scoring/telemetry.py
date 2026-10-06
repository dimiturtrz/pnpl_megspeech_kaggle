import logging

import pytest
import torch

from megspeech.data.labels.vocabulary import VOCAB_SIZE
from megspeech.evaluation.scoring import telemetry as module
from megspeech.evaluation.scoring.evaluator import Evaluator
from megspeech.evaluation.scoring.score import Epoch
from megspeech.evaluation.scoring.telemetry import Telemetry, TelemetryConfig
from tests.unit.megspeech.evaluation.scoring.evaluator import CPU, Constant, split

PRIOR = torch.zeros(VOCAB_SIZE)


def evaluator() -> Evaluator:
    return Evaluator(CPU, min_novel=1)


def base() -> Epoch:
    return evaluator().report(Constant(), {"a": split(seed=1)}, PRIOR, 0, 0.0)


def made() -> Telemetry:
    return Telemetry.of(evaluator(), TelemetryConfig(), split(n=24), {"a": split(seed=1)})


def test_of() -> None:
    on = made()
    assert on.fitted is not None
    assert on.shuffled is not None
    assert list(on.shuffled) == ["a"]
    off = Telemetry.of(evaluator(), TelemetryConfig(fit=False, neural=False), split(), {"a": split(seed=1)})
    assert off.fitted is None
    assert off.shuffled is None


@pytest.mark.parametrize("words", [1, 5, 12])
def test_subset(words: int) -> None:
    """Spans are drawn until they hold the words asked for; the draw is the seed's alone and leaves the global RNG."""
    train = split(n=24)
    state = torch.get_rng_state()
    chosen = Telemetry.subset(train, words, 0)
    assert torch.equal(torch.get_rng_state(), state)
    assert len(chosen.heard()) >= words
    assert len(chosen.index) <= len(train.index)
    assert torch.equal(chosen.index, Telemetry.subset(train, words, 0).index)
    assert torch.equal(chosen.y, train.y)


def test_measured(caplog: pytest.LogCaptureFixture) -> None:
    row = base()
    with caplog.at_level(logging.INFO, logger=module.__name__):
        scored = made().measured(Constant(), row, PRIOR)
        unchanged = Telemetry(evaluator()).measured(Constant(), row, PRIOR)
    assert scored.fit is not None
    assert scored.shuffled is not None
    assert scored.macro == row.macro
    assert unchanged == row
    assert len(caplog.records) == 1
    assert "neural" in caplog.records[0].getMessage()


def test_fit() -> None:
    value = made().fit(Constant(), base(), PRIOR)
    assert value is not None
    assert 0.0 <= value <= 1.0
    assert Telemetry(evaluator()).fit(Constant(), base(), PRIOR) is None


def test_neural() -> None:
    """A model that ignores MEG scores the same on shuffled windows as on real ones."""
    row = base()
    value = made().neural(Constant(), row, PRIOR)
    assert value == pytest.approx(row.macro.by_tau[row.tau])
    assert Telemetry(evaluator()).neural(Constant(), row, PRIOR) is None
