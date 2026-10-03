import math

import torch

from megspeech.model.intervals import HOLDOUT_GAP, Intervals

MASK = torch.tensor([[True, True, True, False]])
ONSET = torch.tensor([[0.0, HOLDOUT_GAP, 3 * HOLDOUT_GAP, 0.0]])


def test_width() -> None:
    assert Intervals.NONE.width == 0
    assert Intervals.LOG.width == len(Intervals.LOG.features(ONSET, MASK)[0, 0])


def test_features() -> None:
    """Gaps 0.21 s then 0.42 s, a padded tail: logs 0 and log 2, the tail and the edges flagged unknown."""
    assert Intervals.NONE.features(ONSET, MASK).shape == (1, 4, 0)
    log2 = math.log(2)
    expected = torch.tensor([[[0.0, 1, 0, 0], [log2, 1, 0, 1], [0, 0, log2, 1], [0, 0, 0, 0]]])
    assert torch.allclose(Intervals.LOG.features(ONSET, MASK), expected, atol=1e-6)
