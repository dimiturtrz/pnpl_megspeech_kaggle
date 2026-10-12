import math

import torch

from megspeech.model.context.intervals import ENDING, Intervals, Reader
from megspeech.model.context.timing import FEATURES, KAGGLE_GAP, TimingHead

MASK = torch.tensor([[True, True, True, False]])
ONSET = torch.tensor([[0.0, KAGGLE_GAP, 3 * KAGGLE_GAP, 0.0]])


def test_width() -> None:
    assert Intervals.NONE.width == 0
    assert Intervals.LOG.width == len(Intervals.LOG.features(ONSET, MASK)[0, 0])
    assert Intervals.RICH.width == FEATURES
    assert Intervals.ENDED.width == FEATURES + ENDING


def test_intervals_features() -> None:
    """Gaps 0.21 s then 0.42 s, a padded tail: logs 0 and log 2, the tail and the edges flagged unknown."""
    assert Intervals.NONE.features(ONSET, MASK).shape == (1, 4, 0)
    log2 = math.log(2)
    expected = torch.tensor([[[0.0, 1, 0, 0], [log2, 1, 0, 1], [0, 0, log2, 1], [0, 0, 0, 0]]])
    assert torch.allclose(Intervals.LOG.features(ONSET, MASK), expected, atol=1e-6)


FINISH = torch.tensor([4 * KAGGLE_GAP])
"""The last word, heard at 3 gaps, ends one gap later."""


def test_ended_features() -> None:
    """With no finish every end is unknown; with one, the span's end reaches every word."""
    assert Intervals.ENDED.features(ONSET, MASK)[..., FEATURES + 1].eq(0).all()
    assert Intervals.ENDED.features(ONSET, MASK, FINISH)[0, :3, FEATURES + 1].tolist() == [1.0, 1.0, 1.0]


def test_ending() -> None:
    """Four gaps to the end from the first word, one from the last (its duration), pace 3 words per 4 gaps; an end at
    or before the last onset is unknown, and padding is zero."""
    features = Reader(ONSET, MASK, FINISH).ending()
    assert features.shape == (1, 4, ENDING)
    assert torch.allclose(features[0, :3, 0], torch.tensor([math.log(4), math.log(3), 0.0]), atol=1e-5)
    assert torch.allclose(features[0, :3, 2], torch.full((3,), math.log(0.75)), atol=1e-5)
    assert features[0, 3].eq(0).all()
    assert Reader(ONSET, MASK, torch.tensor([3 * KAGGLE_GAP])).ending().eq(0).all()


def test_none() -> None:
    assert Reader(ONSET, MASK, FINISH).none().shape == (1, 4, 0)


def test_log() -> None:
    assert torch.equal(Reader(ONSET, MASK, FINISH).log(), Intervals.LOG.features(ONSET, MASK))


def test_rich() -> None:
    """The onsets alone: the finish changes nothing."""
    assert torch.equal(Reader(ONSET, MASK, FINISH).rich(), TimingHead.features(ONSET, MASK))


def test_ended() -> None:
    ended = Reader(ONSET, MASK, FINISH).ended()
    assert torch.equal(ended[..., :FEATURES], TimingHead.features(ONSET, MASK))
    assert torch.equal(ended[..., FEATURES:], Reader(ONSET, MASK, FINISH).ending())
