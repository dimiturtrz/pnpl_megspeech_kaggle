import math

import pytest
import torch

from megspeech.objective.auxiliary.contrastive.similarity import Similarity


def test_forward() -> None:
    x = torch.randn(2, 3, 4, generator=torch.Generator().manual_seed(0))
    assert Similarity().forward(x) is x


def test_scale() -> None:
    similarity = Similarity()
    assert similarity.scale(torch.tensor(8)).item() == pytest.approx(math.log(8))
    assert similarity.scale(torch.tensor(1)).item() == pytest.approx(math.log(2))
