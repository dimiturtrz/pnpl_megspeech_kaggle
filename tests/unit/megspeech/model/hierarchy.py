import pytest
import torch
from torch import nn

from megspeech.data.labels.context_tags import CLASSES
from megspeech.data.labels.phonemes import UNKNOWN
from megspeech.data.labels.vocabulary import IGNORE, VOCAB_SIZE
from megspeech.model.config import ContextConfig
from megspeech.model.hierarchy import ADMIT, Factorised, Flat, Hierarchy

WIDTH = 6
GENERATOR = torch.Generator().manual_seed(0)


def logits() -> torch.Tensor:
    return torch.randn(2, 3, VOCAB_SIZE, generator=GENERATOR)


def refined() -> torch.Tensor:
    return torch.randn(2, 3, WIDTH, generator=GENERATOR)


def sure(head: Factorised, tag: int) -> Factorised:
    """`head` with a tag head certain of `tag` whatever it reads."""
    bias = torch.full((len(head.admitted),), -50.0)
    bias[tag] = 50.0
    head.tag.load_state_dict({"1.weight": torch.zeros(len(head.admitted), WIDTH), "1.bias": bias})
    return head


def test_of() -> None:
    assert isinstance(Hierarchy.of(ContextConfig()), Flat)
    built = Hierarchy.of(ContextConfig(width=WIDTH, hierarchical=True))
    assert isinstance(built, Factorised)
    assert built.readout()(torch.zeros(1, WIDTH)).shape == (1, len(CLASSES))


def test_untagged() -> None:
    with pytest.raises(ValueError, match="tags"):
        Hierarchy.untagged()


def test_flat_forward() -> None:
    raw = logits()
    assert Flat().forward(raw, refined()) is raw
    assert not list(Flat().state_dict())


def test_factorised_forward() -> None:
    """A tag set admitting every word everywhere leaves the head's own log-softmax, whatever the tags say."""
    raw = logits()
    out = Factorised(WIDTH, 0.0).forward(raw, refined())
    assert torch.allclose(out, raw.log_softmax(-1), atol=1e-5)


def test_forward_follows_a_sure_tag() -> None:
    """A certain tag scores its own words by their softmax among themselves; the rest fall far behind."""
    head = Factorised(WIDTH, 0.0, tags=2)
    head.admitted[0, VOCAB_SIZE // 2 :] = False
    head.admitted[1, : VOCAB_SIZE // 2] = False
    raw = logits()
    out = sure(head, 0)(raw, refined())
    assert torch.allclose(out[..., : VOCAB_SIZE // 2], raw[..., : VOCAB_SIZE // 2].log_softmax(-1), atol=1e-4)
    assert bool((out[..., VOCAB_SIZE // 2 :] < -50).all())
    assert torch.allclose(out.logsumexp(-1), torch.zeros(2, 3), atol=1e-5)


def test_forward_drops_tags_admitting_none_of_the_50() -> None:
    """A sure tag that admits no word leaves the mixture of the others, renormalised, with no NaN."""
    head = Factorised(WIDTH, 0.0, tags=2)
    head.admitted[1] = False
    out = sure(head, 1)(logits(), refined())
    assert bool(torch.isfinite(out).all())
    assert torch.allclose(out.logsumexp(-1), torch.zeros(2, 3), atol=1e-5)


def test_within() -> None:
    head = Factorised(WIDTH, 0.0, tags=2)
    head.admitted[1, 1:] = False
    got = head.within(torch.randn(4, VOCAB_SIZE, generator=GENERATOR))
    assert got.shape == (4, 2, VOCAB_SIZE)
    assert torch.allclose(got[:, 1, 0], torch.zeros(4))
    assert bool(torch.isneginf(got[:, 1, 1:]).all())


def test_readout() -> None:
    head = Factorised(WIDTH, 0.0)
    assert head.readout() is head.tag


def test_flat_tagging() -> None:
    word_head = nn.Linear(WIDTH, VOCAB_SIZE)
    assert Flat().tagging(word_head) is word_head


def test_factorised_tagging() -> None:
    head = Factorised(WIDTH, 0.0)
    assert head.tagging(nn.Linear(WIDTH, VOCAB_SIZE)) is head


def test_affinity() -> None:
    """Shares per word over its labelled, tagged tokens; out-of-50 and untagged rows count for nothing."""
    y = torch.tensor([0, 0, 0, 1, IGNORE, 2])
    tags = torch.tensor([1, 1, 0, 2, 0, UNKNOWN])
    shares = Factorised.affinity(y, tags, 3)
    assert torch.allclose(shares[:, 0], torch.tensor([1 / 3, 2 / 3, 0.0]))
    assert torch.allclose(shares[:, 1], torch.tensor([0.0, 0.0, 1.0]))
    assert torch.allclose(shares[:, 2], torch.zeros(3))


def test_factorised_admit() -> None:
    """A tag admits a word at a share of `ADMIT` or more; a word never tagged is admitted by every tag."""
    rare = round(1 / ADMIT) * 2
    y = torch.tensor([0] * rare + [1] * 2)
    tags = torch.tensor([0] * (rare - 1) + [1] + [2, 0])
    head = Factorised(WIDTH, 0.0, tags=3)
    head.admit(y, lambda: tags)
    assert head.admitted[:, 0].tolist() == [True, False, False]
    assert head.admitted[:, 1].tolist() == [True, False, True]
    assert bool(head.admitted[:, 2:].all())
    with pytest.raises(ValueError, match="tags for"):
        head.admit(y, lambda: tags[:1])


def test_flat_admit() -> None:
    flat = Flat()
    flat.admit(torch.zeros(3, dtype=torch.long), Hierarchy.untagged)
    assert not list(flat.state_dict())
