from dataclasses import replace

import numpy as np
import torch
from torch import nn

from megspeech.data.batching.split import DENSE, Split
from megspeech.data.constants import CHANNELS
from megspeech.data.labels.context_tags import CLASSES
from megspeech.data.labels.vocabulary import VOCAB_SIZE
from megspeech.data.spans.spans import Spans
from megspeech.evaluation.scoring.evaluator import Evaluator
from megspeech.evaluation.scoring.score import Head
from megspeech.model.auxiliary import Tap
from megspeech.model.config import ContextConfig
from megspeech.model.hierarchy import Factorised
from megspeech.model.sentence import SentenceNet
from megspeech.model.words.conv import ConvConfig
from megspeech.objective.auxiliary.tagging import TagTerm
from megspeech.objective.term import ContextTerm
from megspeech.training.auxiliary import Joint

FLAT = ContextConfig(width=16, heads=2, layers=1, conv=ConvConfig(hidden=8, dilations=(1, 2)))
FACTORISED = replace(FLAT, hierarchical=True)
CPU = torch.device("cpu")
WORDS = 12
SAMPLES = 50


def split() -> Split:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(WORDS, CHANNELS, SAMPLES)).astype(np.float32)
    onset = np.r_[np.arange(DENSE) * 0.2, 10 + np.arange(WORDS - DENSE) * 2.0]
    return Split.from_arrays(x, np.arange(WORDS) % 3, Spans.by_gap(onset), CPU)


def tags() -> torch.Tensor:
    return torch.arange(WORDS) % len(CLASSES)


def test_off_is_the_flat_model() -> None:
    """Off, the model holds no hierarchy state and its context scores are the context head's own; on, it draws the
    flat model's parameters first, so the two share every weight the flat one has."""
    torch.manual_seed(0)
    flat = SentenceNet(nn.Identity(), FLAT).eval()
    torch.manual_seed(0)
    factorised = SentenceNet(nn.Identity(), FACTORISED).eval()
    assert not [key for key in flat.state_dict() if key.startswith("hierarchy")]
    shared = factorised.state_dict()
    assert all(torch.equal(value, shared[key]) for key, value in flat.state_dict().items())
    held = split()
    with torch.no_grad():
        inputs = held.batch(torch.arange(len(held.index)), WORDS).inputs()
        refined, _, context, _ = flat.heard(inputs)
        assert torch.equal(context, flat.head(refined) + flat.timing(inputs.onset, inputs.mask))


def test_factorised_head_trains_and_scores() -> None:
    """A factorised model admitted from its training words' tags, beside the tag auxiliary at the context tap: the
    tag term trains the hierarchy's own tag head, the context loss reaches it too, and evaluation reads
    normalised log-probabilities."""
    held = split()
    torch.manual_seed(0)
    net = SentenceNet(nn.Identity(), FACTORISED)
    net.hierarchy.admit(held.y, tags)
    hierarchy = net.hierarchy
    assert isinstance(hierarchy, Factorised)
    assert int(hierarchy.admitted.sum()) == len(set(zip(tags().tolist(), held.y.tolist(), strict=True))) + (
        VOCAB_SIZE - len(set(held.y.tolist()))
    ) * len(CLASSES)
    term = TagTerm(1.0, tags())
    model = Joint((term,), Tap.CONTEXT).trained(net, lambda m: m)
    assert model.heads[0] is hierarchy.tag
    batch = held.batch(torch.arange(len(held.index)), WORDS)
    context, _, tagged = model(batch.inputs())
    target = batch.target(held.y)
    plain = ContextTerm(0.0, None)
    loss = plain.loss(plain.read(net.head, context, target)) + term.loss(
        term.read(model.heads[0], tagged, batch.target(term.labels(held)))
    )
    loss.backward()
    linear = hierarchy.tag[1]
    assert isinstance(linear, nn.Linear)
    assert linear.weight.grad is not None
    assert float(linear.weight.grad.abs().sum()) > 0
    scores = Evaluator(CPU).sweep(net.eval(), held)[Head.CONTEXT]
    assert scores.shape == (WORDS, VOCAB_SIZE)
    assert torch.allclose(scores.logsumexp(-1), torch.zeros(WORDS), atol=1e-4)
