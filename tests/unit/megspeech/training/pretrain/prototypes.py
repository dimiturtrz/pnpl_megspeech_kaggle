import math

import pytest
import torch

from megspeech.training.pretrain.prototypes import Assignment, Centring, Prototypes, Scores


def test_validates() -> None:
    with pytest.raises(ValueError, match="prototypes"):
        Prototypes(k=1)
    with pytest.raises(ValueError, match="temp_warmup"):
        Prototypes(temp_warmup=1.5)
    with pytest.raises(ValueError, match="centre_momentum"):
        Prototypes(centre_momentum=1.0)
    assert Prototypes().centring is Centring.EMA


def test_teacher_temp() -> None:
    config = Prototypes(teacher_temp_start=0.04, teacher_temp_end=0.07, temp_warmup=0.5)
    assert config.teacher_temp(0.0) == 0.04
    assert config.teacher_temp(0.25) == pytest.approx(0.055)
    assert config.teacher_temp(0.5) == pytest.approx(0.07)
    assert config.teacher_temp(1.0) == pytest.approx(0.07)
    assert Prototypes(temp_warmup=0.0).teacher_temp(0.0) == Prototypes().teacher_temp_end


def test_sinkhorn() -> None:
    torch.manual_seed(0)
    logits = torch.randn(40, 8)
    probs = Scores(logits, 0.1).sinkhorn(3)
    assert probs.shape == logits.shape
    assert torch.allclose(probs.sum(dim=1), torch.ones(40), atol=1e-5)
    assert probs.sum(dim=0).std() < Scores(logits, 0.1).sinkhorn(0).sum(dim=0).std()
    skewed = logits.clone()
    skewed[:, 0] += 2.0
    assert Scores(skewed, 0.5).sinkhorn(50).sum(dim=0).std() < 0.01
    assert Scores(skewed, 0.5).sinkhorn(0).sum(dim=0).std() > 1.0


def test_cross_entropy() -> None:
    probs = torch.full((3, 4), 0.25)
    assert Scores(torch.zeros(3, 4), 0.1).cross_entropy(probs).item() == pytest.approx(math.log(4))
    sharp = Scores(torch.tensor([[10.0, 0.0, 0.0, 0.0]]), 1.0)
    one_hot = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    assert sharp.cross_entropy(one_hot) < 1e-3
    assert sharp.cross_entropy(one_hot.roll(1, 1)) > 5.0


def test_entropy() -> None:
    assert Assignment(torch.full((5, 4), 0.25)).entropy().item() == pytest.approx(math.log(4), abs=1e-5)
    assert Assignment(torch.eye(4)).entropy().item() == pytest.approx(0.0, abs=1e-5)


def test_perplexity() -> None:
    assert Assignment(torch.eye(4)).perplexity().item() == pytest.approx(4.0, rel=1e-4)
    assert Assignment(torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 6)).perplexity().item() == pytest.approx(1.0, rel=1e-4)
