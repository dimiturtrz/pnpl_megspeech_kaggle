from typing import override

import pytest
import torch
from jaxtyping import Bool, Float
from torch import Tensor

from megspeech.data.labels.vocabulary import VOCAB_SIZE
from megspeech.model.words.base import WordEncoder

SAMPLES = 50
STEPS = 7
GENERATOR = torch.Generator().manual_seed(0)


class Pooled(WordEncoder):
    def __init__(self, *, instance: bool = False) -> None:
        super().__init__(5, instance=instance)
        self.head = self.classifier(0.0)

    @override
    def embed(self, x: Float[Tensor, "b c t"], valid: Bool[Tensor, " b"] | None = None) -> Float[Tensor, "b e"]:
        return x.mean(-1)


class Timed(Pooled):
    @override
    def time_width(self) -> int:
        return 5

    @override
    def embed_timed(
        self, x: Float[Tensor, "b c t"], valid: Bool[Tensor, " b"] | None = None
    ) -> tuple[Float[Tensor, "b e"], Float[Tensor, "b h s"]]:
        return x.mean(-1), x[..., :STEPS]


def test_classifier() -> None:
    assert Pooled().classifier(0.5)(torch.zeros(2, 5)).shape == (2, VOCAB_SIZE)


def test_attach() -> None:
    encoder = Pooled()
    encoder.attach(3)
    assert encoder.classify(torch.zeros(2, 4, 5), torch.zeros(2, 4, 3)).shape == (2, 4, VOCAB_SIZE)


def test_classify() -> None:
    encoder, slots = Pooled(), torch.randn(2, 4, 5, generator=GENERATOR)
    assert torch.equal(encoder.classify(slots, torch.zeros(2, 4, 3)), encoder.head(slots))


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


def test_positions() -> None:
    encoder = Pooled()
    x = torch.randn(3, 5, SAMPLES, generator=GENERATOR)
    slot = torch.tensor([[0, 2, 3], [1, 3, 3]])
    out = encoder.positions(x, torch.zeros(2, 3), slot)
    assert out.shape == (2, 3, 5)
    assert torch.equal(out[0, 1], x[2].mean(-1))
    assert torch.all(out[slot == 3] == 0)


def test_forward() -> None:
    encoder = Pooled()
    x = torch.randn(3, 5, SAMPLES, generator=GENERATOR)
    assert torch.equal(encoder.forward(x), encoder.head(x.mean(-1)))


def test_time_width() -> None:
    assert Timed().time_width() == 5
    with pytest.raises(NotImplementedError, match="Pooled"):
        Pooled().time_width()


def test_embed_timed() -> None:
    x = torch.randn(3, 5, SAMPLES, generator=GENERATOR)
    embedded, times = Timed().embed_timed(x)
    assert torch.equal(embedded, Pooled().embed(x))
    assert times.shape == (3, 5, STEPS)
    with pytest.raises(NotImplementedError, match="Pooled"):
        Pooled().embed_timed(x)


def test_positions_timed() -> None:
    """The embeddings are `positions`' own, each position's time course its row's, zeros where it is empty."""
    encoder = Timed()
    x = torch.randn(3, 5, SAMPLES, generator=GENERATOR)
    slot = torch.tensor([[0, 2, 3], [1, 3, 3]])
    embedded, times = encoder.positions_timed(x, slot)
    assert torch.equal(embedded, encoder.positions(x, torch.zeros(2, 3), slot))
    assert times.shape == (2, 3, 5, STEPS)
    assert torch.equal(times[0, 1], x[2, :, :STEPS])
    assert torch.all(times[slot == 3] == 0)


class Tapped(Pooled):
    @override
    def tap_widths(self) -> tuple[int, ...]:
        return (5,)

    @override
    def embed_tapped(
        self, x: Float[Tensor, "b c t"], taps: tuple[int, ...], levels: tuple[int, ...], valid: Bool[Tensor, " b"]
    ) -> tuple[Float[Tensor, "b e"], Float[Tensor, "b k"]]:
        return x.mean(-1), x[:, :, : len(taps) * len(levels)].flatten(1)


def test_tap_widths() -> None:
    assert Tapped().tap_widths() == (5,)
    with pytest.raises(NotImplementedError, match="Pooled"):
        Pooled().tap_widths()


def test_embed_tapped() -> None:
    x = torch.randn(3, 5, SAMPLES, generator=GENERATOR)
    valid = torch.ones(3, dtype=torch.bool)
    embedded, pooled = Tapped().embed_tapped(x, (0,), (1, 2), valid)
    assert torch.equal(embedded, Pooled().embed(x))
    assert pooled.shape == (3, 10)
    with pytest.raises(NotImplementedError, match="Pooled"):
        Pooled().embed_tapped(x, (0,), (1,), valid)


def test_positions_tapped() -> None:
    encoder = Tapped()
    x = torch.randn(3, 5, SAMPLES, generator=GENERATOR)
    slot = torch.tensor([[0, 2, 3], [1, 3, 3]])
    embedded, pooled = encoder.positions_tapped(x, slot, (0,), (1,))
    assert torch.equal(embedded, encoder.positions(x, torch.zeros(2, 3), slot))
    assert torch.equal(pooled[0, 1], x[2, :, :1].flatten())
    assert torch.all(pooled[slot == 3] == 0)


def test_occupied() -> None:
    x = torch.zeros(3, 5, SAMPLES)
    assert WordEncoder.occupied(x, torch.tensor([[0, 2, 3]])).tolist() == [True, False, True]


def test_placed() -> None:
    rows = torch.arange(6.0).reshape(2, 3)
    placed = WordEncoder.placed(rows, torch.tensor([[1, 2], [0, 2]]))
    assert placed.tolist() == [[[3.0, 4.0, 5.0], [0.0, 0.0, 0.0]], [[0.0, 1.0, 2.0], [0.0, 0.0, 0.0]]]


def test_core() -> None:
    encoder = Pooled()
    assert encoder.core() == (encoder,)
