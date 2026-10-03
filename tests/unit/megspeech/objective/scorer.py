import math

import pytest
import torch

from megspeech.objective.scorer import BIAS, TEMPERATURE, SigmoidScorer


def test_score() -> None:
    """A row parallel to one anchor and orthogonal to the other scores `exp(temperature) + bias` and `bias`."""
    scorer = SigmoidScorer(4, 3)
    anchors = torch.eye(3)[:2]
    logits = scorer.score(torch.tensor([[5.0, 0.0, 0.0]]), anchors)
    assert logits.shape == (1, 2)
    assert logits[0, 0].item() == pytest.approx(math.exp(TEMPERATURE) + BIAS)
    assert logits[0, 1].item() == pytest.approx(BIAS)
