import math
from dataclasses import dataclass
from typing import override

import torch
from torch import nn

from megspeech.data.batching.split import Split
from megspeech.data.constants import CHANNELS
from megspeech.data.labels.vocabulary import IGNORE, VOCAB_SIZE
from megspeech.data.spans.spans import PAD, Spans
from megspeech.evaluation.studies.fusion.masked import MaskedPrior, Specials

SPECIALS = Specials(cls=1, sep=2, mask=3, pad=0)
IDS = 100 + torch.arange(VOCAB_SIZE)
TOKENS = 200
INDEX = torch.tensor([[0, 1, 2], [3, PAD, PAD]])
Y = torch.tensor([5, IGNORE, 7, 9])
GUESS = torch.tensor([10, 11, 12, 13])
CPU = torch.device("cpu")


@dataclass(frozen=True)
class Output:
    logits: torch.Tensor


class Ramp(nn.Module):
    @override
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Output:
        return Output(
            torch.arange(TOKENS, dtype=torch.float).expand(*input_ids.shape, TOKENS) * attention_mask[..., None]
        )


def test_queries() -> None:
    rows, at, word = MaskedPrior(Ramp(), IDS, SPECIALS).queries(INDEX, Y, GUESS)
    m, c, s, p = SPECIALS.mask, SPECIALS.cls, SPECIALS.sep, SPECIALS.pad
    expected = torch.tensor([[c, m, m, 112, s], [c, 110, m, m, s], [c, m, s, p, p]])
    assert torch.equal(rows, expected)
    assert at.tolist() == [1, 3, 1]
    assert word.tolist() == [0, 2, 3]


def test___call__() -> None:
    out = MaskedPrior(Ramp(), IDS, SPECIALS)(INDEX, Y, GUESS)
    assert out.shape == (len(Y), VOCAB_SIZE)
    assert torch.allclose(out.logsumexp(-1), torch.zeros(len(Y)), atol=1e-5)
    assert torch.all(out[[0, 2, 3]].argmax(dim=1) == VOCAB_SIZE - 1)
    assert torch.allclose(out[1], torch.full((VOCAB_SIZE,), -math.log(VOCAB_SIZE)))


def test_logp() -> None:
    """The split's words read with the context head's guesses filling the span."""
    onset = torch.tensor([0.0, 0.2, 0.4, 5.0]).numpy()
    split = Split.from_arrays(
        torch.zeros(len(onset), CHANNELS, 4).numpy(), Y.clamp(min=0).numpy(), Spans.by_gap(onset), CPU
    )
    context = torch.zeros(len(onset), VOCAB_SIZE).scatter_(1, GUESS[:, None], 1.0)
    prior = MaskedPrior(Ramp(), IDS, SPECIALS)
    assert torch.equal(prior.logp(split, context), prior(split.index, split.y, GUESS))
