from typing import override

import torch
from jaxtyping import Bool, Float
from torch import Tensor

from megspeech.data.labels.vocabulary import VOCAB_SIZE
from megspeech.model.words.base import WordEncoder

SAMPLES = 50
GENERATOR = torch.Generator().manual_seed(0)


class Pooled(WordEncoder):
    def __init__(self, *, instance: bool = False) -> None:
        super().__init__(5, instance=instance)
        self.head = self.classifier(0.0)

    @override
    def embed(self, x: Float[Tensor, "b c t"], valid: Bool[Tensor, " b"] | None = None) -> Float[Tensor, "b e"]:
        return x.mean(-1)


def test_classifier() -> None:
    assert Pooled().classifier(0.5)(torch.zeros(2, 5)).shape == (2, VOCAB_SIZE)


def test_normalise() -> None:
    x = 3.0 + 2.0 * torch.randn(4, 5, SAMPLES, generator=GENERATOR)
    present = torch.ones(4, 1, SAMPLES, dtype=torch.bool)
    present[0, :, 40:] = False
    assert torch.equal(Pooled().normalise(x, present), x)
    normalised = Pooled(instance=True).normalise(x, present)
    assert torch.all(normalised[0, :, 40:] == 0)
    kept = normalised[0, :, :40]
    assert torch.allclose(kept.mean(-1), torch.zeros(5), atol=1e-5)
    assert torch.allclose(kept.std(-1, unbiased=False), torch.ones(5), atol=1e-4)
    assert torch.allclose(normalised[1:].std(-1, unbiased=False), torch.ones(3, 5), atol=1e-4)


def test_standardise() -> None:
    flat = WordEncoder.standardise(torch.full((1, 2, SAMPLES), 7.0), torch.ones(1, 1, SAMPLES, dtype=torch.bool))
    assert torch.all(flat == 0)


def test_forward() -> None:
    encoder = Pooled()
    x = torch.randn(3, 5, SAMPLES, generator=GENERATOR)
    assert torch.equal(encoder.forward(x), encoder.head(x.mean(-1)))
