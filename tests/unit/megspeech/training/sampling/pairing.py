import numpy as np
import pandas as pd
import pytest
import torch

from megspeech.data.batching.split import Split
from megspeech.data.spans.spans import Spans
from megspeech.training.sampling.pairing import Pairing, Unpaired

CPU = torch.device("cpu")
WORDS = pd.DataFrame(
    {
        "run": [
            "sub-101_ses-0_task-MASC1_run-1",
            "sub-101_ses-0_task-MASC1_run-1",
            "sub-102_ses-0_task-MASC1_run-1",
            "sub-102_ses-0_task-MASC1_run-1",
            "sub-101_ses-1_task-MASC1_run-1",
            "sub-0_ses-1_task-Sherlock1_run-1",
        ],
        "position": [0, 1, 0, 1, 0, 0],
    }
)
"""One word a span: spans 0, 2 and 4 hear one token (4 is subject 101's other session), 1 and 3 another, 5 none."""
PARTNERS = {0: {2}, 1: {3}, 2: {0, 4}, 3: {1}, 4: {2}}


def split(n: int = len(WORDS)) -> Split:
    return Split.from_arrays(np.zeros((n, 4, 4), np.float32), np.arange(n) % 3, Spans.by_gap(np.arange(n) * 9.0), CPU)


def pairing(share: float = 1.0, batch: int = 4) -> Pairing:
    made = Pairing.of(share, WORDS, split(), batch)
    assert isinstance(made, Pairing)
    return made


def test_unpaired_drawn() -> None:
    draws = torch.arange(8)
    assert Unpaired().drawn(draws, torch.ones(6), 4) is draws


def test_of() -> None:
    made = pairing(0.5, 8)
    assert made.anchors == 2
    assert made.paired.tolist() == [True, True, True, True, True, False]
    assert isinstance(Pairing.of(0.0, WORDS, split(), 8), Unpaired)
    with pytest.raises(ValueError, match="no training span"):
        Pairing.of(1.0, WORDS.iloc[[0, 1, 4, 5]].reset_index(drop=True), split(4), 4)


def test_blocks() -> None:
    order = np.array([2, 0, 1])
    start, size = Pairing.blocks(np.array([[5], [5], [7]]), order)
    assert start.tolist() == [0, 2, 0]
    assert size.tolist() == [2, 1, 2]


def test_partners() -> None:
    made = pairing()
    torch.manual_seed(0)
    spans = torch.tensor([0, 1, 2, 3, 4] * 40)
    found = made.partners(spans)
    for span, partner in zip(spans.tolist(), found.tolist(), strict=True):
        assert partner in PARTNERS[span]
    assert set(found[spans == 2].tolist()) == {0, 4}


def test_pairing_drawn() -> None:
    """Each batch opens with its anchors, then their partners; the rest of the batch is left as drawn."""
    made = pairing(0.5, 8)
    torch.manual_seed(0)
    draws = torch.full((24,), 5)
    batches = made.drawn(draws, torch.ones(6), 8).view(3, 8)
    for row in batches.tolist():
        assert row[2] in PARTNERS[row[0]] and row[3] in PARTNERS[row[1]]
        assert row[4:] == [5, 5, 5, 5]
    assert draws.tolist() == [5] * 24
