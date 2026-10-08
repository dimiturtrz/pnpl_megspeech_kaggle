import math
from pathlib import Path
from typing import override

import numpy as np
import pytest
import torch
from torch import nn

from megspeech.data.batching.split import SpanInput
from megspeech.data.constants import CHANNELS, RATE
from megspeech.data.labels.vocabulary import PRIMARY, VOCAB_SIZE
from megspeech.evaluation.studies.attribution import (
    Attribution,
    AttributionConfig,
    Drops,
    Reads,
    Saliency,
    Topomap,
)
from tests.unit.megspeech.evaluation.scoring.evaluator import split


class Probe(nn.Module):
    """Reads the first sensor's window sum and the onset: every word's logits are those two scalars times fixed
    per-word coefficients, so attribution can only land on sensor 0."""

    def __init__(self) -> None:
        super().__init__()
        self.coef = nn.Parameter(torch.linspace(-1, 1, VOCAB_SIZE))

    @override
    def forward(self, spans: SpanInput) -> tuple[torch.Tensor, torch.Tensor]:
        score = nn.functional.pad(spans.x[:, 0].sum(-1), (0, 1))[spans.slot]
        logits = score[..., None] * self.coef + spans.onset[..., None] * self.coef.flip(0)
        return logits, logits


def attribution(words: int = 6, steps: int = 4) -> Attribution:
    return Attribution(Probe(), {"d": split(n=12)}, frozenset(), AttributionConfig(steps=steps, words=words, chunk=2))


def first() -> Reads:
    return attribution().reads()[0]


def test_objective() -> None:
    read = first()
    x, onset = read.copies(5)
    out = read.objective(x, onset)
    assert out.shape == (5,)
    batch = len(read.own)
    assert torch.allclose(out[:batch], out[batch : 2 * batch])
    assert out.max() <= 0


def test_reads_of() -> None:
    domain = split(n=12)
    words = domain.novel(frozenset())[:2]
    span = int(Attribution.spans(domain, words).popitem()[0])
    read = Reads.of(Probe(), domain, span, words[:1])
    assert read.x.shape[1:] == (CHANNELS, 50)
    assert len(read.own) == 1
    assert read.objective.labels.tolist() == [int(domain.y[words[0]])]


def test_copies() -> None:
    x, onset = first().copies(3)
    assert x.shape[0] == onset.shape[0] == 3


def saliency() -> Saliency:
    magnitude = torch.zeros(4, 3, 5)
    magnitude[:, 1, 2] = torch.tensor([1.0, 1.0, 2.0, 4.0])
    return Saliency(magnitude, torch.tensor([7, 7, 3, 7]))


def test_share() -> None:
    assert Saliency.share(torch.tensor([1.0, 3.0])).tolist() == [0.25, 0.75]
    assert Saliency.share(torch.zeros(3)).tolist() == [0.0, 0.0, 0.0]


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


def test_spans() -> None:
    domain = split(n=12)
    words = domain.novel(frozenset())
    by_span = Attribution.spans(domain, words)
    assert sorted(torch.cat(list(by_span.values())).tolist()) == sorted(words.tolist())
    for span, group in by_span.items():
        assert set(group.tolist()) <= set(domain.index[span].tolist())


def test_reads() -> None:
    reads = attribution().reads()
    assert sum(len(read.own) for read in reads) == 4
    assert all(len(read.own) <= 2 for read in reads)


def test_integrated() -> None:
    """The probe reads sensor 0 alone, so every other sensor's attribution is exactly zero."""
    read = first()
    magnitude = attribution().integrated(read)
    assert magnitude.shape == (len(read.own), CHANNELS, 50)
    assert magnitude[:, 1:].sum() == 0
    assert magnitude[:, 0].sum() > 0


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
