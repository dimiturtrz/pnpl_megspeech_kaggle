from typing import override

import torch
from torch import nn

from megspeech.data.batching.split import SpanInput, Split
from megspeech.data.constants import CHANNELS
from megspeech.data.labels.vocabulary import VOCAB_SIZE
from megspeech.evaluation.saliency import Integrated, Reads
from tests.unit.megspeech.evaluation.scoring.evaluator import split


class Probe(nn.Module):
    """Reads the first sensor's window sum and the onset: every word's logits are those two scalars times fixed
    per-word coefficients, so attribution can only land on sensor 0."""

    def __init__(self) -> None:
        super().__init__()
        self.coef = nn.Parameter(torch.linspace(-1, 1, VOCAB_SIZE))

    @override
    def forward(self, spans: SpanInput) -> tuple[torch.Tensor, torch.Tensor]:
        score = nn.functional.pad(spans.x[:, 0].sum(-1), (0, 1))[spans.slot]
        logits = score[..., None] * self.coef + spans.onset[..., None] * self.coef.flip(0)
        return logits, logits


def integrated() -> Integrated:
    return Integrated(Probe(), steps=4, chunk=2)


def scored() -> tuple[Split, torch.Tensor]:
    domain = split(n=12)
    return domain, domain.novel(frozenset())[:4]


def first() -> Reads:
    return integrated().reads(*scored())[0]


def test_objective() -> None:
    read = first()
    x, onset = read.copies(5)
    out = read.objective(x, onset)
    assert out.shape == (5,)
    batch = len(read.own)
    assert torch.allclose(out[:batch], out[batch : 2 * batch])
    assert out.max() <= 0


def test_of() -> None:
    domain, words = scored()
    span = int(Integrated.spans(domain, words[:1]).popitem()[0])
    read = Reads.of(Probe(), domain, span, words[:1])
    assert read.x.shape[1:] == (CHANNELS, 50)
    assert len(read.own) == 1
    assert read.words.tolist() == words[:1].tolist()
    assert read.objective.labels.tolist() == [int(domain.y[words[0]])]


def test_copies() -> None:
    x, onset = first().copies(3)
    assert x.shape[0] == onset.shape[0] == 3


def test_share() -> None:
    assert Integrated.share(torch.tensor([1.0, 3.0])).tolist() == [0.25, 0.75]
    assert Integrated.share(torch.zeros(3)).tolist() == [0.0, 0.0, 0.0]


def test_spans() -> None:
    domain = split(n=12)
    words = domain.novel(frozenset())
    by_span = Integrated.spans(domain, words)
    assert sorted(torch.cat(list(by_span.values())).tolist()) == sorted(words.tolist())
    for span, group in by_span.items():
        assert set(group.tolist()) <= set(domain.index[span].tolist())


def test_reads() -> None:
    reads = integrated().reads(*scored())
    assert sum(len(read.own) for read in reads) == 4
    assert all(len(read.own) <= 2 for read in reads)


def test_magnitude() -> None:
    """The probe reads sensor 0 alone, so every other sensor's attribution is exactly zero."""
    read = first()
    magnitude = integrated().magnitude(read)
    assert magnitude.shape == (len(read.own), CHANNELS, 50)
    assert magnitude[:, 1:].sum() == 0
    assert magnitude[:, 0].sum() > 0


def test_sensors() -> None:
    """Rows follow the asked order, each a share landing wholly on the one sensor the probe reads."""
    domain, words = scored()
    shares = integrated().sensors(domain, words.flip(0))
    assert shares.shape == (len(words), CHANNELS)
    assert torch.allclose(shares[:, 0], torch.ones(len(words)))
    first_word = integrated().sensors(domain, words[-1:])
    assert torch.allclose(shares[0], first_word[0])
