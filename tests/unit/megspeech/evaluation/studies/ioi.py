import numpy as np
import pytest
import torch

from megspeech.data.batching.split import Split
from megspeech.data.spans.spans import Spans
from megspeech.evaluation.metric.ruler import Graded
from megspeech.evaluation.studies.ioi import Replacement, Timing

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


def test_positions() -> None:
    assert timing().positions() == "0:0.150(n=2) 1:0.300(n=1)"


def test_replaced() -> None:
    """Onsets are rebuilt from the new gaps, each span starting at zero."""
    gaps = torch.tensor([[1.0, 2.0, 0, 0, 0, 0, 0, 0]] * 3)
    onset = timing().replaced(gaps).onset
    assert onset[0, :3].tolist() == [0.0, 1.0, 3.0]
    assert onset.shape == (3, 9)


def test_replacement_gaps() -> None:
    """Every replacement fills the pairs only; a shuffle keeps each span's own gaps."""
    base = timing()
    real, pair, pool = base.gaps(), base.pair(), torch.tensor([0.5, 0.7])
    for replacement in Replacement:
        new = replacement.gaps(real, pair, pool, torch.Generator().manual_seed(0))
        assert (new[~pair] == 0).all()
        assert (new[pair] > 0).all()
    shuffled = Replacement.SHUFFLE.gaps(real, pair, pool, torch.Generator().manual_seed(0))
    assert sorted(shuffled[0, :2].tolist()) == pytest.approx([0.1, 0.3])
    assert Replacement.CONSTANT.gaps(real, pair, pool, torch.Generator()).unique().tolist() == [0.0, 0.5]


def test_shuffled() -> None:
    base = timing()
    out = Replacement.shuffled(base.gaps(), base.pair(), torch.ones(1), torch.Generator().manual_seed(1))
    assert out[1, 0] == pytest.approx(0.2)


def test_sampled() -> None:
    base = timing()
    out = Replacement.sampled(base.gaps(), base.pair(), torch.tensor([0.5]), torch.Generator())
    assert out.unique().tolist() == [0.5]


def test_median() -> None:
    base = timing()
    out = Replacement.median(base.gaps(), base.pair(), torch.tensor([0.2, 0.4, 0.9]), torch.Generator())
    assert out.unique().tolist() == pytest.approx([0.4])


def test_score() -> None:
    domains = {"d": (torch.tensor([True, False, True, True]), torch.tensor([0, 1, 2, 3]))}
    assert "n=4" in Timing.score(Graded(domains))
    assert Timing.score(Graded(domains)).startswith("macro ")


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
