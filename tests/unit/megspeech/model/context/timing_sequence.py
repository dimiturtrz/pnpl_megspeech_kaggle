import torch

from megspeech.model.context.timing import KAGGLE_GAP
from megspeech.model.context.timing_sequence import TimingSequence

ONSET = torch.tensor([[0.0, KAGGLE_GAP, 3 * KAGGLE_GAP, 0.0]])
MASK = torch.tensor([[True, True, True, False]])


def test_forward() -> None:
    """One logit row per position, and a word's logits move when a far word's gap does: the span is read whole."""
    torch.manual_seed(0)
    model = TimingSequence(5, layers=1).eval()
    logits = model.forward(ONSET, MASK)
    moved = model(torch.tensor([[0.0, KAGGLE_GAP, 9 * KAGGLE_GAP, 0.0]]), MASK)
    assert logits.shape == (1, 4, 5)
    assert not torch.allclose(logits[0, 0], moved[0, 0])
