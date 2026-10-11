import math

import numpy as np
import pytest
import torch
from torch import nn

from megspeech.data.batching.split import Split
from megspeech.data.constants import CHANNELS
from megspeech.data.labels.context_tags import CLASSES, OTHER
from megspeech.data.spans.spans import Spans
from megspeech.model.words.tagged import TaggedHead
from megspeech.objective.auxiliary.tagging import Tagging, TagTerm

CPU = torch.device("cpu")
WIDTH = 6
TAGS = torch.tensor([3, OTHER, -1])


def term() -> TagTerm:
    return TagTerm(2.0, TAGS)


def test_tagging() -> None:
    assert isinstance(TaggedHead(WIDTH, 0.0), Tagging)
    assert not isinstance(nn.Linear(WIDTH, 2), Tagging)


def test_head() -> None:
    with pytest.raises(ValueError, match="tagged"):
        term().head(WIDTH)


def test_readout() -> None:
    head = TaggedHead(WIDTH, 0.0)
    shared = term().readout(head, WIDTH)
    assert shared is head.tag
    with pytest.raises(ValueError, match="tagged"):
        term().readout(nn.Linear(WIDTH, 2), WIDTH)


def test_count() -> None:
    assert term().count() == len(CLASSES)


def split() -> Split:
    return Split.from_arrays(
        np.zeros((3, CHANNELS, 4), np.float32), np.zeros(3, np.int64), Spans.by_gap(np.arange(3.0)), CPU
    )


def test_labels() -> None:
    assert term().labels(split()).tolist() == TAGS.tolist()
    with pytest.raises(ValueError, match="tags for"):
        TagTerm(1.0, TAGS[:2]).labels(split())


def test_classify() -> None:
    assert term().classify(torch.tensor([[3, -1, -100]])).tolist() == [[3, -1, -1]]


def test_unweighted() -> None:
    """Uniform logits cost log 12 per tagged row; untagged and empty rows cost nothing."""
    tagged = term()
    uniform = torch.zeros(1, 3, len(CLASSES))
    target = torch.tensor([[3, OTHER, -100]])
    assert tagged.loss(tagged.read(nn.Identity(), uniform, target)).item() == pytest.approx(
        2.0 * math.log(len(CLASSES))
    )
    assert tagged.unweighted(tagged.read(nn.Identity(), uniform, torch.tensor([[-1, -100, -100]]))).item() == 0.0
