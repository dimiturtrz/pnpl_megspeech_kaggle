import torch

from megspeech.data.batching.split import Split
from megspeech.data.constants import CHANNELS
from megspeech.data.spans.spans import Spans
from megspeech.evaluation.studies.timefusion import Expert
from megspeech.model.context.timing_only import TimingOnly

CPU = torch.device("cpu")
ROWS = 6


def test_words() -> None:
    """Every spanned word gets a normalised row over the vocabulary, in the split's word order."""
    onset = torch.arange(ROWS, dtype=torch.float64).numpy() * 0.2 + (torch.arange(ROWS) // 3).numpy() * 5.0
    split = Split.from_arrays(
        torch.zeros(ROWS, CHANNELS, 4).numpy(), (torch.arange(ROWS) % 3).numpy(), Spans.by_gap(onset), CPU
    )
    words = Expert(TimingOnly(7, layers=1).eval()).words(split)
    assert words.shape == (ROWS, 7)
    assert torch.allclose(words.exp().sum(-1), torch.ones(ROWS))
