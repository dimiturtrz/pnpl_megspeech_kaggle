import math

import torch

from megspeech.data.labels.vocabulary import VOCAB_SIZE
from megspeech.model.context.timing import FEATURES, KAGGLE_GAP, Clock, Jitter, TimingHead

MASK = torch.tensor([[True, True, True, False]])
ONSET = torch.tensor([[0.0, KAGGLE_GAP, 3 * KAGGLE_GAP, 0.0]])


def test_jitter_apply() -> None:
    """Sigma 0 is the identity; otherwise onsets move, stay ordered, start at zero and leave the padding alone."""
    assert Jitter().apply(ONSET, MASK) is ONSET
    torch.manual_seed(0)
    moved = Jitter(0.5).apply(ONSET, MASK)
    assert not torch.equal(moved, ONSET)
    assert moved[0, 0] == 0
    assert (moved[0, 1:3] >= moved[0, :2]).all()
    assert moved[0, 3] == 0


def test_jitter_ordered() -> None:
    """Moved onsets that cross are sorted among the present words; the padded slot keeps its old value."""
    moved = torch.tensor([[1.0, 0.5, 2.0, 9.0]])
    assert Jitter.ordered(moved, MASK, ONSET).tolist() == [[0.0, 0.5, 1.5, 0.0]]


def test_times() -> None:
    onset = torch.tensor([[0.0, 0.5, 0.0]])
    mask = torch.tensor([[True, True, False]])
    assert torch.equal(Clock.ONSET.times(onset, mask), onset)
    assert torch.allclose(Clock.ORDER.times(onset, mask)[0, :2], torch.tensor([0.0, KAGGLE_GAP]))


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


def test_timing_head_forward() -> None:
    """Active, logits at each position; inactive, a scalar 0 and nothing saved."""
    assert TimingHead(5).forward(ONSET, MASK).shape == (1, 4, 5)
    inactive = TimingHead(5, active=False)
    assert inactive.forward(ONSET, MASK).shape == ()
    assert not inactive.state_dict()


def test_over_vocabulary() -> None:
    assert TimingHead.over_vocabulary().forward(ONSET, MASK).shape == (1, 4, VOCAB_SIZE)
    assert not TimingHead.over_vocabulary(active=False).state_dict()


def test_frozen() -> None:
    """A trained head's weights are loaded and no gradient reaches them."""
    donor = TimingHead(5)
    held = TimingHead(5)
    held.frozen(donor.state_dict())
    assert torch.equal(held.forward(ONSET, MASK), donor.forward(ONSET, MASK))
    assert not any(parameter.requires_grad for parameter in held.parameters())
