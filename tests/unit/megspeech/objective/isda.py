import pytest
import torch
from torch import nn
from torch.nn import functional

from megspeech.data.labels.vocabulary import IGNORE, VOCAB_SIZE
from megspeech.objective.isda import Isda, Unaugmented
from megspeech.objective.term import Reading, WordTerm

FEATURES = torch.tensor([[[1.0, 0.0], [3.0, 0.0], [0.0, 1.0], [5.0, 5.0]]])
TARGET = torch.tensor([[0, 0, 1, IGNORE]])


def head() -> nn.Sequential:
    """Class 0 reads the first feature, class 1 the second, every other class neither."""
    linear = nn.Linear(2, VOCAB_SIZE)
    with torch.no_grad():
        linear.weight.zero_()
        linear.bias.zero_()
        linear.weight[0, 0] = linear.weight[1, 1] = 1.0
    return nn.Sequential(nn.Dropout(0.0), linear)


def isda(strength: float = 0.5) -> tuple[nn.Sequential, Isda]:
    readout = head()
    augmentation = Isda.of(strength, readout)
    assert isinstance(augmentation, Isda)
    return readout, augmentation


def test_of() -> None:
    assert type(Isda.of(0.0, head())) is Unaugmented
    readout, augmentation = isda()
    assert augmentation.linear is readout[-1]
    assert augmentation.covariance.shape == (VOCAB_SIZE, 2, 2)
    with pytest.raises(TypeError, match="linear word head"):
        Isda.of(0.5, nn.Sequential(nn.Linear(2, VOCAB_SIZE), nn.ReLU()))


def test_isda_seen() -> None:
    """Training features are kept for the loss; an evaluation read leaves them."""
    readout, augmentation = isda()
    readout(FEATURES)
    assert augmentation.features is not None
    assert torch.equal(augmentation.features, FEATURES)
    readout.eval()
    readout(2 * FEATURES)
    assert torch.equal(augmentation.features, FEATURES)


def test_isda_ramp() -> None:
    """`lambda0 * epoch / epochs`, the paper's schedule."""
    _, augmentation = isda(0.5)
    assert augmentation.ramp(0, 10) == 0.0
    assert augmentation.ramp(4, 10) == pytest.approx(0.2)
    assert augmentation.scale == pytest.approx(0.2)


def test_unaugmented_ramp() -> None:
    assert Unaugmented().ramp(4, 10) == 0.0


def test_isda_drained() -> None:
    _, augmentation = isda(0.5)
    augmentation.ramp(5, 10)
    assert augmentation.drained() == pytest.approx(0.25)


def test_unaugmented_drained() -> None:
    assert Unaugmented().drained() == 0.0


def test_update() -> None:
    """Two batches merged at their counts give the covariance of their union; `IGNORE` rows count nowhere."""
    _, augmentation = isda()
    first = torch.tensor([[1.0, 0.0], [3.0, 2.0], [9.0, 9.0]])
    second = torch.tensor([[5.0, 1.0], [0.0, 4.0]])
    augmentation.update(first, torch.tensor([0, 0, IGNORE]))
    augmentation.update(second, torch.tensor([0, 2]))
    union = torch.cat([first[:2], second[:1]])
    assert torch.allclose(augmentation.mean[0], union.mean(0))
    assert torch.allclose(augmentation.covariance[0], torch.cov(union.T, correction=0))
    assert augmentation.count[:3].tolist() == [3.0, 0.0, 1.0]
    assert augmentation.covariance[2].abs().sum() == 0


def test_spread() -> None:
    """Class 0 varies along the first feature alone (variance 1): every rival's direction from it spans that
    feature with weight 1, so each quadratic form is 1; class 1 does not vary, so its row is 0."""
    _, augmentation = isda()
    augmentation.update(FEATURES[0], TARGET[0])
    spread = augmentation.spread(augmentation.linear.weight)
    assert spread.shape == (VOCAB_SIZE, VOCAB_SIZE)
    assert spread[0, 0] == 0
    assert torch.allclose(spread[0, 1:], torch.ones(VOCAB_SIZE - 1))
    assert spread[1].abs().sum() == 0


def test_isda_augmented() -> None:
    """The closed form `z_j + (lambda / 2) (w_j - w_y)^T Sigma_y (w_j - w_y)`, evaluated by hand, at the ramped
    strength; the `IGNORE` row is left as read."""
    readout, augmentation = isda(0.5)
    augmentation.ramp(1, 1)
    logits = readout(FEATURES)
    expected = logits.clone()
    expected[0, :2, 1:] += 0.25
    assert torch.allclose(augmentation.augmented(logits, TARGET), expected)


def test_unaugmented_augmented() -> None:
    logits = head()(FEATURES)
    assert Unaugmented().augmented(logits, TARGET) is logits


def test_augmented_unseen() -> None:
    readout, augmentation = isda()
    readout.eval()
    with pytest.raises(RuntimeError, match="no training features"):
        augmentation.augmented(readout(FEATURES), TARGET)


def test_strength_zero_is_plain_cross_entropy() -> None:
    """At lambda 0 the augmented loss is the plain smoothed cross-entropy, bit for bit."""
    readout, augmentation = isda()
    augmentation.ramp(0, 10)
    logits = readout(FEATURES)
    reading = Reading(readout, logits, TARGET, torch.ones(TARGET.shape))
    plain = WordTerm(0.1, None).unweighted(reading)
    assert torch.equal(WordTerm(0.1, None, isda=augmentation).unweighted(reading), plain)
    expected = functional.cross_entropy(
        logits.flatten(0, 1), TARGET.flatten(), ignore_index=IGNORE, label_smoothing=0.1
    )
    assert torch.equal(plain, expected)
