import torch

from megspeech.model.context.timing import KAGGLE_GAP
from megspeech.model.context.timing_only import TimingOnly

ONSET = torch.tensor([[0.0, KAGGLE_GAP, 3 * KAGGLE_GAP, 0.0]])


def test_forward() -> None:
    """Both heads read the one timing logit per position; a position with no window is masked."""
    context, word = TimingOnly(5).forward(
        torch.zeros(1, 1, 1), ONSET, torch.tensor([[0, 1, 2, 3]]), torch.zeros(1, dtype=torch.long)
    )
    assert context.shape == (1, 4, 5)
    assert context is word
