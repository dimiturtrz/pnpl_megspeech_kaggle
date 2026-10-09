import numpy as np
import pytest
import torch

from megspeech.data.batching.split import Split
from megspeech.data.batching.windows import Windows
from megspeech.data.constants import CHANNELS
from megspeech.data.labels.vocabulary import IGNORE
from megspeech.data.spans.spans import MAX_GAP, PAD, Spans
from megspeech.training.regularise.rightcut import RightCut, Whole

RATE = 10.0
SHARE = 0.3
MIXED = 0.26
"""Between the mixed spans' uncut share, 0.227, and their all-cut share, 0.294."""


def spoken(labels: list[int], onsets: list[float]) -> Split:
    """Words at `onsets` seconds read at 10 Hz, each bounded at its span's next onset, a span's last 2 s on."""
    at = torch.tensor([round(onset * RATE) for onset in onsets])
    joined = torch.from_numpy(np.diff(onsets) <= MAX_GAP)
    following = torch.cat([torch.where(joined, at[1:], at[:-1] + 20), at[-1:] + 20])
    windows = Windows(torch.zeros(CHANNELS, int(at[-1]) + 80), at, following, following=following)
    words = np.array(labels)
    return Split.of(windows, (words, words), Spans.by_gap(np.array(onsets)), np.ones(len(labels))).finished(RATE)


def always() -> RightCut:
    return RightCut(1.0, RATE, torch.zeros(4))


def test_of() -> None:
    train = spoken([0, 1, 2, 3], [0.0, 0.5, 1.0, 1.5])
    assert type(RightCut.of(train, 0.0)) is Whole
    made = RightCut.of(train, SHARE, RATE)
    assert isinstance(made, RightCut)
    assert made.p == pytest.approx(4 / 9)
    assert made.rate == RATE


def test_chance() -> None:
    """One span of four labelled words: uncut a quarter end it, cut ones keep 2.5 on average, so share 1/(4 - 1.5p)."""
    index = torch.tensor([[0, 1, 2, 3]])
    labelled = torch.ones(4, dtype=torch.bool)
    weight = torch.ones(1)
    p, natural = RightCut.chance(index, labelled, weight, SHARE)
    assert (p, natural) == (pytest.approx(4 / 9), pytest.approx(0.25))
    assert RightCut.chance(index, labelled, weight, 0.2)[0] == 0.0
    assert RightCut.chance(index, labelled, weight, 0.9)[0] == 1.0
    assert RightCut.chance(torch.tensor([[0, 1]]), labelled, weight, SHARE)[0] == 0.0


def test_chance_matches_drawn_share() -> None:
    """The solved probability lands its share on spans of mixed lengths with unlabelled words among them."""
    torch.manual_seed(0)
    lengths = [2, 3, 5, 7, 9]
    onsets = [float(10 * span + word) * 0.5 for span, length in enumerate(lengths) for word in range(length)]
    labels = [IGNORE if word % 4 == 3 else 1 for length in lengths for word in range(length)]
    train = spoken(labels, onsets)
    made = RightCut.of(train, MIXED, RATE)
    assert isinstance(made, RightCut)
    assert 0.0 < made.p < 1.0
    spans = torch.arange(len(train.index))
    for _ in range(4000):
        made.apply(train.batch(spans, len(train.y)), train)
    assert made.drained()["share"] == pytest.approx(MIXED, abs=0.005)


def test_right_cut_apply() -> None:
    """A three-word span always cut keeps its first two words; a two-word span is never cut."""
    train = spoken([0, 1, 2, 3, 4], [0.0, 0.5, 1.0, 9.0, 9.5])
    batch = train.batch(torch.tensor([0, 1]), 5)
    short = always().apply(batch, train)
    assert short.index[:, :3].tolist() == [[0, 1, PAD], [3, 4, PAD]]
    assert short.slot[:, :3].tolist() == [[0, 1, 5], [3, 4, 5]]
    assert short.target(train.y)[:, :3].tolist() == [[0, 1, IGNORE], [3, 4, IGNORE]]
    assert torch.equal(short.x, batch.x)


def test_whole_apply() -> None:
    train = spoken([0, 1, 2], [0.0, 0.5, 1.0])
    batch = train.batch(torch.tensor([0]), 3)
    assert Whole().apply(batch, train) is batch


def test_finished() -> None:
    """A cut span ends at its kept last word's next onset; an uncut one keeps its epoch's end."""
    train = spoken([0, 1, 2, 3, 4], [0.0, 0.5, 1.0, 9.0, 9.5])
    batch = train.batch(torch.tensor([0, 1]), 5)
    short = always().apply(batch, train)
    assert batch.finish is not None
    assert batch.finish.tolist() == pytest.approx([2.0, 1.5])
    assert short.finish is not None
    assert short.finish.tolist() == pytest.approx([1.0, 1.5])
    signal = np.zeros((3, CHANNELS, 5), np.float32)
    plain = Split.from_arrays(signal, np.zeros(3, np.int64), Spans.by_gap(np.zeros(3)), torch.device("cpu"))
    unknown = plain.batch(torch.tensor([0]), 3)
    assert always().finished(unknown, unknown.index, torch.tensor([True]), plain) is None


def test_count() -> None:
    train = spoken([0, IGNORE, 2, 3, 4], [0.0, 0.5, 1.0, 9.0, 9.5])
    made = always()
    short = made.apply(train.batch(torch.tensor([0, 1]), 5), train)
    assert made.tally.tolist() == [1.0, 2.0, 1.0, 3.0]
    made.count(short, torch.tensor([False, False]), train)
    assert made.tally.tolist() == [1.0, 4.0, 2.0, 6.0]


def test_right_cut_drained() -> None:
    train = spoken([0, 1, 2], [0.0, 0.5, 1.0])
    made = always()
    made.apply(train.batch(torch.tensor([0]), 3), train)
    assert made.drained() == {"cut": 1.0, "spans": 1.0, "last": 1.0, "labelled": 2.0, "share": 0.5}
    assert made.drained()["spans"] == 0.0


def test_whole_drained() -> None:
    assert Whole().drained() == {}
