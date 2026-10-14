import argparse
import logging
from typing import cast

import numpy as np
import pytest
import torch

from megspeech.data.batching.split import Split
from megspeech.data.raw.runs import Partition
from megspeech.data.spans.spans import Spans
from megspeech.evaluation.loaded.model import LoadedModel
from megspeech.evaluation.metric.ruler import Graded
from megspeech.evaluation.studies.ioi import Held, Mode, Redraw, Replacement, Scored, Timing
from tests.unit.megspeech.evaluation.studies.gap import Stub, split

CPU = torch.device("cpu")
ONSETS = np.array([0.0, 0.1, 0.4, 10.0, 10.2, 20.0])
GROUPS = np.array([0, 0, 0, 1, 1, 2])


def timing() -> Timing:
    spans = Spans.by_sentence(GROUPS, ONSETS)
    return Timing(Split.from_arrays(np.zeros((6, 4, 4), dtype=np.float32), np.arange(6), spans, CPU))


def test_pair() -> None:
    """Both neighbours exist inside a span; a one-word span and the padding hold none."""
    assert timing().pair().sum(dim=1).tolist() == [2, 1, 0]


def test_timing_gaps() -> None:
    assert timing().gaps()[0, :2].tolist() == pytest.approx([0.1, 0.3])
    assert timing().gaps()[2].abs().sum() == 0


def test_values() -> None:
    assert timing().values().tolist() == pytest.approx([0.1, 0.3, 0.2])


def test_summary() -> None:
    assert "n=3" in timing().summary()
    assert "alone=0.333" in timing().summary()


def test_timing_grid() -> None:
    """Gaps 0.1, 0.3, 0.2 s all sit on the 250 Hz and millisecond grids; the minimum is 100 ms."""
    report = timing().grid()
    assert "on-1/250=1.000" in report
    assert "on-1/1000=1.000" in report
    assert "min=100.0ms" in report


def test_timing_positions() -> None:
    assert timing().positions() == "0:0.150(n=2) 1:0.300(n=1)"


def test_places() -> None:
    """Spans of three, two and one word: places from the start and from the end, by word id."""
    start, end = timing().places(torch.arange(6))
    assert start.tolist() == [0, 1, 2, 0, 1, 0]
    assert end.tolist() == [2, 1, 0, 1, 0, 0]


def test_buckets() -> None:
    """Each word is its own class; the last bucket pools every later place; an empty bucket is NaN."""
    start, end = torch.tensor([0, 1, 2, 3]), torch.tensor([3, 2, 1, 0])
    hits, labels = torch.tensor([True, False, True, True]), torch.arange(4)
    report = Timing.buckets(start, end, hits, labels)
    assert "last=1.0000 n=1 second-last=1.0000 n=1 third-last=0.0000 n=1 earlier=1.0000 n=1" in report
    assert "first=1.0000 n=1 second=0.0000 n=1 rest=1.0000 n=2" in report
    assert "nan n=0" in Timing.buckets(start, end * 0 + 3, hits, labels)


def test_positioned() -> None:
    base = timing()
    graded = Graded({"d": (torch.tensor([True] * 5), torch.arange(5))})
    report = Scored(graded).positioned({"d": base.split}, frozenset())
    assert "last=1.0000 n=2 second-last=1.0000 n=2 third-last=1.0000 n=1 earlier=nan n=0" in report


def test_replaced() -> None:
    """Onsets are rebuilt from the new gaps, each span starting at zero."""
    gaps = torch.tensor([[1.0, 2.0, 0, 0, 0, 0, 0, 0]] * 3)
    onset = timing().replaced(gaps).onset
    assert onset[0, :3].tolist() == [0.0, 1.0, 3.0]
    assert onset.shape == (3, 9)


def redraw(pool: torch.Tensor, seed: int = 0) -> Redraw:
    base = timing()
    return Redraw(base.gaps(), base.pair(), pool, torch.Generator().manual_seed(seed))


def test_replacement_gaps() -> None:
    """Every replacement fills the pairs only; a shuffle keeps each span's own gaps."""
    pair = timing().pair()
    for replacement in Replacement:
        new = replacement.gaps(redraw(torch.tensor([0.5, 0.7])))
        assert (new[~pair] == 0).all()
        assert (new[pair] > 0).all()
    shuffled = Replacement.SHUFFLE.gaps(redraw(torch.tensor([0.5, 0.7])))
    assert sorted(shuffled[0, :2].tolist()) == pytest.approx([0.1, 0.3])
    assert Replacement.CONSTANT.gaps(redraw(torch.tensor([0.5, 0.7]))).unique().tolist() == [0.0, 0.5]


def test_shuffled() -> None:
    assert redraw(torch.ones(1), seed=1).shuffled()[1, 0] == pytest.approx(0.2)


def test_sampled() -> None:
    assert redraw(torch.tensor([0.5])).sampled().unique().tolist() == [0.5]


def test_median() -> None:
    assert redraw(torch.tensor([0.2, 0.4, 0.9])).median().unique().tolist() == pytest.approx([0.4])


def test_redrawn() -> None:
    """The constant draw rebuilds every span at the pool's median spacing."""
    onset = timing().redrawn(Replacement.CONSTANT, torch.tensor([0.5]), torch.Generator()).onset
    assert onset[0, :3].tolist() == pytest.approx([0.0, 0.5, 1.0])
    assert onset[2, 1] == 0


def test_score() -> None:
    domains = {"d": (torch.tensor([True, False, True, True]), torch.tensor([0, 1, 2, 3]))}
    assert "n=4" in Scored(Graded(domains)).score()
    assert Scored(Graded(domains)).score().startswith("macro ")


def test_blurred() -> None:
    """Words stay in order and each span restarts at zero; padding is untouched."""
    base = timing()
    onset = base.split.onset.clone()
    onset[0, :3] = torch.tensor([0.5, 0.3, 0.9])
    out = base.blurred(onset).onset
    assert out[0, :3].tolist() == pytest.approx([0.0, 0.2, 0.6])
    assert out[2, 0] == 0


def test_jittered() -> None:
    base = timing()
    out = base.jittered(0.01, torch.Generator().manual_seed(0)).onset
    assert (out[0, 1:3] - out[0, :2] > 0).all()
    assert out[0, 0] == 0
    assert not torch.equal(out, base.split.onset)


def test_quantised() -> None:
    out = timing().quantised(0.25).onset
    assert out[0, :3].tolist() == [0.0, 0.0, 0.5]


def test_blurs() -> None:
    arms = Timing.blurs(0)
    assert list(arms)[:2] == ["real", "jitter5ms"]
    assert len(arms) == 8
    base = timing()
    assert arms["real"](base) is base.split
    assert arms["quant20ms"](base).onset.shape == base.split.onset.shape


class Partitioned(Stub):
    partition = Partition.TEST


def held() -> Held:
    domains = {"d": split()}
    return Held(cast(LoadedModel, Partitioned()), domains, 0.5, timing(), 0)


def test_chosen() -> None:
    flags = {"grid": False, "positions": True, "blur": True}
    assert Mode.chosen(argparse.Namespace(**flags)) is Mode.POSITIONS
    assert Mode.chosen(argparse.Namespace(**{**flags, "grid": True})) is Mode.GRID
    assert Mode.chosen(argparse.Namespace(grid=False, positions=False, blur=False)) is Mode.GAPS


def test_held_graded() -> None:
    """A model reading its own window finds every word of an untouched domain."""
    assert held().graded(held().domains).score().startswith("macro 1.0000")


def test_held_report(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        held().report(Mode.GRID)
    assert caplog.messages[0].startswith("kaggle-test gaps")
    assert caplog.messages[1].startswith("test d gaps")


def test_held_grid(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        held().grid()
    assert len(caplog.messages) == 2
    assert "on-1/250" in caplog.messages[1]


def test_held_positions(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        held().positions()
    assert caplog.messages == [caplog.messages[0]]
    assert caplog.messages[0].startswith("test tau 0.5 balanced top-10")


def test_held_blur(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        held().blur()
    assert [message.split()[1] for message in caplog.messages] == [f"onsets={label}" for label in Timing.blurs(0)]


def test_held_gaps(caplog: pytest.LogCaptureFixture) -> None:
    """The statistics first, then the real gaps and every replacement in order, a word reader blind to all."""
    with caplog.at_level(logging.INFO):
        held().gaps()
    scored = [message for message in caplog.messages if " gaps=" in message]
    assert [message.split()[1] for message in scored] == ["gaps=real", *(f"gaps={r}" for r in Replacement)]
    assert all("macro 1.0000" in message for message in scored)
