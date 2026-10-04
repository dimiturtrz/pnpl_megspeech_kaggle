import math

import torch

from megspeech.model.context.timing import FEATURES, HOLDOUT_GAP, Clock, Intervals, TimingHead

MASK = torch.tensor([[True, True, True, False]])
ONSET = torch.tensor([[0.0, HOLDOUT_GAP, 3 * HOLDOUT_GAP, 0.0]])


def test_width() -> None:
    assert Intervals.NONE.width == 0
    assert Intervals.LOG.width == len(Intervals.LOG.features(ONSET, MASK)[0, 0])


def test_intervals_features() -> None:
    """Gaps 0.21 s then 0.42 s, a padded tail: logs 0 and log 2, the tail and the edges flagged unknown."""
    assert Intervals.NONE.features(ONSET, MASK).shape == (1, 4, 0)
    log2 = math.log(2)
    expected = torch.tensor([[[0.0, 1, 0, 0], [log2, 1, 0, 1], [0, 0, log2, 1], [0, 0, 0, 0]]])
    assert torch.allclose(Intervals.LOG.features(ONSET, MASK), expected, atol=1e-6)


def test_times() -> None:
    onset = torch.tensor([[0.0, 0.5, 0.0]])
    mask = torch.tensor([[True, True, False]])
    assert torch.equal(Clock.ONSET.times(onset, mask), onset)
    assert torch.allclose(Clock.ORDER.times(onset, mask)[0, :2], torch.tensor([0.0, HOLDOUT_GAP]))


def test_neighbour() -> None:
    """The next word 0.21 s on is log 1 = 0, the one before the first is unknown, none wraps past the span's end."""
    after = TimingHead.neighbour(ONSET, MASK, 1)
    assert torch.allclose(after[0, :, 0], torch.tensor([0.0, math.log(2), 0.0, 0.0]), atol=1e-6)
    assert after[0, :, 1].tolist() == [1.0, 1.0, 0.0, 0.0]
    assert TimingHead.neighbour(ONSET, MASK, -2)[0, :, 1].tolist() == [0.0, 0.0, 1.0, 0.0]


def test_timing_head_features() -> None:
    """Every feature at each word, zeros at padding; a lone word is flagged alone."""
    features = TimingHead.features(ONSET, MASK)
    assert features.shape == (1, 4, FEATURES)
    assert features[0, 3].eq(0).all()
    alone = TimingHead.features(torch.zeros(1, 2), torch.tensor([[True, False]]))
    assert alone[0, 0, -1] == 1.0
    assert torch.isfinite(features).all()


def test_forward() -> None:
    """Active, logits at each position; inactive, a scalar 0 and nothing saved."""
    assert TimingHead(5).forward(ONSET, MASK).shape == (1, 4, 5)
    inactive = TimingHead(5, active=False)
    assert inactive.forward(ONSET, MASK).shape == ()
    assert not inactive.state_dict()
