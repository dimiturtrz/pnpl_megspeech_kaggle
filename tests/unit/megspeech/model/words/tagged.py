import torch

from megspeech.data.labels.context_tags import CLASSES
from megspeech.data.labels.vocabulary import VOCAB_SIZE
from megspeech.model.words.tagged import TaggedHead

WIDTH = 6

GENERATOR = torch.Generator().manual_seed(0)


def test_forward() -> None:
    assert TaggedHead(WIDTH, 0.0).forward(torch.randn(2, 3, WIDTH, generator=GENERATOR)).shape == (2, 3, VOCAB_SIZE)


def test_forward_is_the_tag_mixture() -> None:
    """With one tag the word logit is that tag's readout alone, shifted by the tag's log-probability of 0."""
    head = TaggedHead(WIDTH, 0.0, tags=1)
    x = torch.randn(4, WIDTH, generator=GENERATOR)
    assert torch.allclose(head(x), head.words(x), atol=1e-6)


def test_forward_follows_the_tag() -> None:
    """A tag the head is sure of lets only its own readout through."""
    head = TaggedHead(WIDTH, 0.0, tags=2)
    with torch.no_grad():
        head.tag.load_state_dict({"1.weight": torch.zeros(2, WIDTH), "1.bias": torch.tensor([50.0, -50.0])})
    x = torch.randn(3, WIDTH, generator=GENERATOR)
    assert torch.allclose(head(x), head.words(x)[..., :VOCAB_SIZE], atol=1e-4)


def test_readout() -> None:
    head = TaggedHead(WIDTH, 0.0)
    assert head.readout()(torch.zeros(5, WIDTH)).shape == (5, len(CLASSES))
