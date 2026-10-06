import math

import pytest
import torch

from megspeech.training.pretrain.prototypes import Assignment, Centring, Prototypes


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
    probs = Assignment.sinkhorn(logits, 0.1, 3)
    assert probs.shape == logits.shape
    assert torch.allclose(probs.sum(dim=1), torch.ones(40), atol=1e-5)
    assert probs.sum(dim=0).std() < Assignment.sinkhorn(logits, 0.1, 0).sum(dim=0).std()
    skewed = logits.clone()
    skewed[:, 0] += 2.0
    assert Assignment.sinkhorn(skewed, 0.5, 50).sum(dim=0).std() < 0.01
    assert Assignment.sinkhorn(skewed, 0.5, 0).sum(dim=0).std() > 1.0


def test_cross_entropy() -> None:
    logits = torch.zeros(3, 4)
    probs = torch.full((3, 4), 0.25)
    assert Assignment.cross_entropy(logits, probs, 0.1).item() == pytest.approx(math.log(4))
    sharp = torch.tensor([[10.0, 0.0, 0.0, 0.0]])
    one_hot = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    assert Assignment.cross_entropy(sharp, one_hot, 1.0) < 1e-3
    assert Assignment.cross_entropy(sharp, one_hot.roll(1, 1), 1.0) > 5.0


def test_entropy() -> None:
    assert Assignment.entropy(torch.full((5, 4), 0.25)).item() == pytest.approx(math.log(4), abs=1e-5)
    assert Assignment.entropy(torch.eye(4)).item() == pytest.approx(0.0, abs=1e-5)


def test_perplexity() -> None:
    assert Assignment.perplexity(torch.eye(4)).item() == pytest.approx(4.0, rel=1e-4)
    assert Assignment.perplexity(torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 6)).item() == pytest.approx(1.0, rel=1e-4)
