from dataclasses import replace

import numpy as np
import torch

from megspeech.data.batching.split import Split
from megspeech.data.batching.windows import Windows
from megspeech.data.spans.spans import Spans
from megspeech.training.sampling.strata import Strata

SPANNED = Split.of(
    Windows.stacked(torch.zeros(6, 1, 2)),
    (np.zeros(6, dtype=np.int64), np.zeros(6, dtype=np.int64)),
    Spans(np.array([[0, 1, 2], [3, -1, -1], [4, 5, -1]]), np.array([[0, 0.2, 0.4], [0, 0, 0], [0, 1.0, 0]])),
    np.ones(6),
)
"""Spans of 3 words at 5 words/s, 1 word (rate 0), 2 words at 1 word/s: each the top, bottom and middle tercile."""


def test_stratify() -> None:
    """Subject, then length tercile, then rate tercile; a second subject's strata sit past the first's."""
    assert Strata.stratify(SPANNED).tolist() == [8, 0, 4]
    other = replace(SPANNED, subject=torch.ones(6, dtype=torch.int64))
    assert Strata.stratify(other).tolist() == [17, 9, 13]


def test_binned() -> None:
    assert Strata.binned(torch.tensor([3.0, 1.0, 2.0])).tolist() == [2, 0, 1]
    assert Strata.binned(torch.arange(6.0)).tolist() == [0, 0, 1, 1, 2, 2]


def test_strata_sourced() -> None:
    """Spans of one stratum split by source, numbered from 0; an unknown source leaves the strata as `stratify`."""
    assert Strata.sourced(SPANNED).tolist() == [8, 0, 4]
    sourced = replace(SPANNED, source=torch.tensor([1, 0, 0, 0, 0, 0]))
    assert Strata.sourced(sourced).tolist() == [2, 0, 1]
    split_apart = replace(SPANNED, source=torch.tensor([0, 0, 0, 0, 1, 1]))
    assert Strata.sourced(split_apart).tolist() == [1, 0, 2]
