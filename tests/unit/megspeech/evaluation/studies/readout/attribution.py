import math
from pathlib import Path

import numpy as np
import pytest
import torch

from megspeech.data.constants import CHANNELS, RATE
from megspeech.data.labels.vocabulary import PRIMARY
from megspeech.evaluation.saliency import Integrated, Reads
from megspeech.evaluation.studies.readout.attribution import (
    Attribution,
    AttributionConfig,
    Drops,
    Saliency,
    Topomap,
)
from tests.unit.megspeech.evaluation.saliency import Probe
from tests.unit.megspeech.evaluation.scoring.evaluator import split


def attribution(words: int = 6, steps: int = 4) -> Attribution:
    return Attribution(Probe(), {"d": split(n=12)}, frozenset(), AttributionConfig(steps=steps, words=words, chunk=2))


def first() -> Reads:
    return attribution().reads()[0]


def saliency() -> Saliency:
    magnitude = torch.zeros(4, 3, 5)
    magnitude[:, 1, 2] = torch.tensor([1.0, 1.0, 2.0, 4.0])
    return Saliency(magnitude, torch.tensor([7, 7, 3, 7]))


def test_sensors() -> None:
    assert saliency().sensors().tolist() == [0.0, 1.0, 0.0]


def test_lags() -> None:
    assert saliency().lags().argmax().item() == 2


def test_frequent() -> None:
    assert saliency().frequent(1) == [7]
    assert saliency().frequent(5) == [7, 3]


def test_by_class() -> None:
    assert len(saliency().by_class(7).labels) == 3
    assert saliency().by_class(3).magnitude.sum() == 2


def test_saliency_summary() -> None:
    summary = saliency().summary(["a", "b", "c"], top=1)
    assert summary["words"] == 4
    assert summary["top_sensors"] == ["b"]
    assert summary["peak_lag_s"] == 2 / RATE


def test_timing_to_meg() -> None:
    drops = Drops(-1.0, 2.0, 1.0, 3.0)
    assert drops.timing_to_meg == 0.5
    assert math.isnan(Drops(-1.0, 0.0, 1.0, 3.0).timing_to_meg)


def test_drops_summary() -> None:
    assert Drops(-1.0, 2.0, 1.0, 3.0).summary()["interval_over_meg"] == 0.5


def test_sampled() -> None:
    one = attribution(words=3).sampled()["d"]
    assert len(one) == 3
    assert torch.equal(one, attribution(words=3).sampled()["d"])
    assert len(attribution(words=1000).sampled()["d"]) == len(split(n=12).novel(frozenset()))


def test_integrated() -> None:
    study = attribution()
    assert study.integrated() == Integrated(study.model, 4, 2)


def test_reads() -> None:
    reads = attribution().reads()
    assert sum(len(read.own) for read in reads) == 4
    assert all(len(read.own) <= 2 for read in reads)


def test_ablated() -> None:
    """Ablating a read's inputs to themselves drops nothing; to zeros drops what the label's log-probability held."""
    study, read = attribution(), first()
    x, onset = read.copies(len(read.own))
    assert torch.allclose(study.ablated(read, (x, onset)), torch.zeros(len(read.own)))
    drop = study.ablated(read, (torch.zeros_like(x), onset))
    full = read.objective(x, onset)
    assert torch.allclose(drop, full - read.objective(torch.zeros_like(x), onset), atol=1e-5)


def test_dropped() -> None:
    full, meg, interval, timing = attribution().dropped(first())
    assert full.shape == meg.shape == interval.shape == timing.shape
    assert meg.abs().sum() > 0
    assert timing.abs().sum() > 0


def test_run() -> None:
    result, drops = attribution().run()
    assert len(result.labels) == 4
    assert result.sensors()[0] == pytest.approx(1.0)
    assert drops.full <= 0


def test_report() -> None:
    names = [f"s{i}" for i in range(CHANNELS)]
    report = attribution().report(names)
    assert set(report) == {"config", "sensors", "overall", "classes", "ablation"}
    classes = report["classes"]
    assert isinstance(classes, dict)
    assert set(classes) <= set(PRIMARY.words)
    overall = report["overall"]
    assert isinstance(overall, dict)
    assert overall["top_sensors"][0] == "s0"


def test_save(tmp_path: Path) -> None:
    positions = np.random.default_rng(0).normal(size=(CHANNELS, 3))
    path = Topomap(positions, np.full(CHANNELS, 1 / CHANNELS)).save(tmp_path / "map.png")
    assert path.stat().st_size > 0
