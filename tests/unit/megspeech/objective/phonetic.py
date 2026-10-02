import math

import pytest
import torch

from megspeech.data.phonemes import PHONEMES, UNKNOWN
from megspeech.objective.phonetic import Phonetic

WIDTH = 6
GENERATOR = torch.Generator().manual_seed(0)


def test_forward() -> None:
    assert Phonetic(WIDTH).forward(torch.randn(2, 3, WIDTH, generator=GENERATOR)).shape == (2, 3, len(PHONEMES))


def test_loss() -> None:
    """Uniform logits cost log(39) on each known row; unknown rows are neither counted nor averaged over."""
    logits = torch.zeros(3, len(PHONEMES))
    assert Phonetic.loss(logits, torch.tensor([0, 5, UNKNOWN])).item() == pytest.approx(math.log(len(PHONEMES)))
    sure = torch.full((1, len(PHONEMES)), -50.0)
    sure[0, 4] = 50.0
    assert Phonetic.loss(sure, torch.tensor([4])).item() == pytest.approx(0.0, abs=1e-6)
    assert Phonetic.loss(logits, torch.full((3,), UNKNOWN)).item() == 0.0
