from dataclasses import replace
from typing import cast

import numpy as np
import pandas as pd
import pytest
import torch

from megspeech.data.batching.split import Split
from megspeech.data.spans.spans import Spans
from megspeech.evaluation.loaded import LoadedModel
from megspeech.evaluation.metric.ruler import Graded
from megspeech.evaluation.studies.readout.timing import Agreement, Probe, Readings, Shuffled
from megspeech.model.auxiliary import Tap

CPU = torch.device("cpu")
WORDS = pd.DataFrame({"run": ["a"] * 4, "sentence": [0, 0, 0, 1], "onset": [1.0, 1.5, 3.5, 9.0]})


def readings(seed: int, n: int = 900) -> Readings:
    """Duration and position are linear in the first two embedding columns."""
    generator = np.random.default_rng(seed)
    x = generator.standard_normal((n, 4))
    embeddings = {Tap.WORD: x, Tap.CONTEXT: generator.standard_normal((n, 4))}
    return Readings(embeddings, np.arange(n) % 3, x[:, 0], 2 * x[:, 1])


def test_targets() -> None:
    """Log gap to the next onset of the sentence, NaN for a sentence's last word; position within the sentence."""
    duration, position = Readings.targets(WORDS)
    assert duration[:2].tolist() == pytest.approx([np.log(0.5), np.log(2.0)])
    assert np.isnan(duration[2:]).all()
    assert position.tolist() == [0, 1, 2, 0]


def test_target() -> None:
    one = readings(0)
    assert np.array_equal(one.target("duration"), one.duration)
    assert np.array_equal(one.target("position"), one.position)


def test_readout() -> None:
    assert Probe.readout() is not Probe.readout()


def test_report() -> None:
    """The word tap holds both targets, the context tap is noise, and NaN targets are left out of the count."""
    train = readings(0)
    train = replace(train, duration=np.where(np.arange(900) % 9 == 0, np.nan, train.duration))
    out = Probe(train, readings(1)).report()
    assert out["word duration out-of-fold"][0] > 0.9
    assert out["word position official-val"][0] > 0.9
    assert out["context duration official-val"][0] < 0.2
    assert out["word duration out-of-fold"][1] == 800
    assert out["word duration official-val"][1] == 900


def hits() -> tuple[np.ndarray, np.ndarray]:
    return np.array([True, True, False, False]), np.array([True, False, True, False])


def test_agreement_of() -> None:
    agreement = Agreement.of(*hits())
    assert (agreement.n, agreement.both, agreement.first_only, agreement.second_only) == (4, 1, 1, 1)
    assert agreement.expected == pytest.approx(1.0)


def test_neither() -> None:
    assert Agreement.of(*hits()).neither() == 1


def test_summary() -> None:
    summary = Agreement.of(*hits()).summary()
    assert summary["agreement"] == 0.5
    assert summary["agreement-if-independent"] == 0.5


def split(n: int = 4) -> Split:
    x = np.zeros((n, 4, 4), dtype=np.float32)
    return Split.from_arrays(x, np.arange(n) % 3, Spans.by_gap(np.arange(n) * 2.0), CPU)


def test_pooled() -> None:
    domains = {"x": (torch.tensor([True, False]), torch.tensor([1, 2])), "y": (torch.tensor([True]), torch.tensor([3]))}
    hits, names = Agreement.pooled(Graded(domains))
    assert hits.tolist() == [True, False, True]
    assert names.tolist() == ["x", "x", "y"]


class Recorder:
    def __init__(self) -> None:
        self.seen: dict[str, Split] = {}
        self.evaluator = self
        self.model: object = None
        self.log_prior: object = None

    def calibration(self, tau: float) -> float:
        return tau

    def graded(self, model: object, domains: dict[str, Split], calibration: float) -> float:
        self.seen = domains
        return calibration


def test_graded() -> None:
    loaded = Recorder()
    original = split(6)
    assert Shuffled(cast(LoadedModel, loaded), 0.5).graded({"d": original}, 0) == 0.5
    assert torch.equal(loaded.seen["d"].windows.at, original.permuted(0).windows.at)
