import pytest
import torch
from torch import nn

from megspeech.data.constants import CHANNELS
from megspeech.data.labels.vocabulary import VOCAB_SIZE
from megspeech.model.pyramid import Pyramid, PyramidConfig, Surface
from megspeech.model.words.conv import ConvConfig, ConvWords

SMALL = ConvConfig(hidden=8, dilations=(1, 2, 4))
CONFIG = PyramidConfig(taps=(0, 1), levels=(1, 2), width=6, bound=0.25)
WIDTH = 5
SAMPLES = 40
GENERATOR = torch.Generator().manual_seed(0)
SLOT = torch.tensor([[0, 2, 3], [1, 3, 3]])
ONSET = torch.zeros(2, 3)


def inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.randn(3, CHANNELS, SAMPLES, generator=GENERATOR)
    refined = torch.randn(2, 3, WIDTH, generator=GENERATOR)
    logits = torch.randn(2, 3, VOCAB_SIZE, generator=GENERATOR)
    return x, refined, logits


def pyramid() -> tuple[Pyramid, ConvWords]:
    words = ConvWords(SMALL).eval()
    return Pyramid(CONFIG, words, WIDTH), words


@pytest.mark.parametrize(
    ("config", "match"),
    [
        ({"taps": (1, 0)}, "taps"),
        ({"taps": (-1,)}, "taps"),
        ({"levels": ()}, "levels"),
        ({"levels": (0,)}, "levels"),
        ({"bound": 0.0}, "bound"),
    ],
)
def test_pyramid_config_refuses(config: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        PyramidConfig(**config)  # pyrefly: ignore[bad-argument-type]


def test_active() -> None:
    assert not PyramidConfig().active
    assert CONFIG.active


def test_readout() -> None:
    words = ConvWords(SMALL)
    assert isinstance(PyramidConfig().readout(words, WIDTH), Surface)
    assert isinstance(CONFIG.readout(words, WIDTH), Pyramid)


def test_pyramid_refuses_taps_past_the_encoder() -> None:
    with pytest.raises(ValueError, match="reach past"):
        Pyramid(PyramidConfig(taps=(len(SMALL.dilations),)), ConvWords(SMALL), WIDTH)


def test_surface_positions() -> None:
    """The surface's embeddings are the encoder's own, with no taps."""
    _, words = pyramid()
    x, _, _ = inputs()
    plain, none = Surface().positions(words, x, ONSET, SLOT)
    assert torch.equal(plain, words.positions(x, ONSET, SLOT))
    assert none.shape == (2, 3, 0)


def test_pyramid_positions() -> None:
    """The pyramid's embeddings are the encoder's own beside each tap pooled at every level, zeros where a
    position is empty."""
    model, words = pyramid()
    x, _, _ = inputs()
    slots, taps = model.positions(words, x, ONSET, SLOT)
    assert torch.allclose(slots, words.positions(x, ONSET, SLOT))
    assert taps.shape == (2, 3, SMALL.hidden * 2 * 3)
    assert torch.all(taps[len(x) == SLOT] == 0)


def test_surface_correct() -> None:
    _, refined, logits = inputs()
    context, word = Surface().correct(refined[..., :0], refined, logits, -logits)
    assert context is logits
    assert word is not logits


def test_pyramid_correct() -> None:
    """The zero-initialised readout leaves both heads' logits as they are."""
    model, _ = pyramid()
    _, refined, logits = inputs()
    taps = torch.randn(2, 3, SMALL.hidden * 2 * 3, generator=GENERATOR)
    context, word = model.correct(taps, refined, logits, -logits)
    assert torch.equal(context, logits)
    assert torch.equal(word, -logits)


def test_residual() -> None:
    """The correction starts at zero and, whatever the last layer's weights, never exceeds the bound."""
    model, _ = pyramid()
    _, refined, _ = inputs()
    taps = torch.randn(2, 3, SMALL.hidden * 2 * 3, generator=GENERATOR)
    assert torch.all(model.residual(taps, refined) == 0)
    nn.init.normal_(model.out.weight, std=10.0, generator=GENERATOR)
    norms = model.residual(taps, refined).norm(dim=-1)
    assert torch.all(norms <= CONFIG.bound * (1 + 1e-6))
    assert torch.all(norms > CONFIG.bound / 2)


def test_hold() -> None:
    """Every module beside the pyramid is in evaluation mode with its gradients off; the pyramid is untouched."""
    model, words = pyramid()
    net = nn.Module()
    net.words = words.train()
    net.pyramid = model.train()
    model.hold(net)
    assert not words.training
    assert not any(p.requires_grad for p in words.parameters())
    assert model.training
    assert all(p.requires_grad for p in model.parameters())
    Surface().hold(net)
    assert not words.training


def test_fresh() -> None:
    model, _ = pyramid()
    assert Surface().fresh() == set()
    assert model.fresh() == {f"pyramid.{key}" for key in model.state_dict()}
    assert "pyramid.out.weight" in model.fresh()
