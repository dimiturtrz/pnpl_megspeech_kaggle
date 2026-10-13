import math

import numpy as np
import pandas as pd
import pytest
import torch

from megspeech.data.batching.split import Split
from megspeech.data.labels.vocabulary import IGNORE
from megspeech.data.spans.spans import Spans
from megspeech.objective.auxiliary.instance import Apart, InstanceTerm, Together
from megspeech.objective.auxiliary.similarity import Similarity
from megspeech.objective.term import Reading

CPU = torch.device("cpu")
WORDS = pd.DataFrame(
    {
        "run": [
            "sub-101_ses-0_task-MASC1_run-1",
            "sub-101_ses-0_task-MASC1_run-1",
            "sub-102_ses-0_task-MASC1_run-1",
            "sub-102_ses-0_task-MASC1_run-1",
            "sub-101_ses-1_task-MASC1_run-1",
            "sub-0_ses-1_task-Sherlock1_run-1",
        ],
        "position": [0, 1, 0, 1, 0, 0],
    }
)
"""Rows 0 and 2 (and 4) hear one token, 1 and 3 another; 4 is subject 101's other session; 5 is no MASC token."""
TARGET = torch.tensor([[0, 1, 2], [3, 4, 5], [IGNORE, IGNORE, IGNORE]])


def term(weight: float = 1.0) -> InstanceTerm:
    return InstanceTerm.of(weight, WORDS, CPU)


def embeddings(seed: int = 0) -> torch.Tensor:
    return torch.randn(3, 3, 4, generator=torch.Generator().manual_seed(seed))


def test_forward() -> None:
    x = torch.randn(2, 3, 4)
    assert Similarity().forward(x) is x


def test_scale() -> None:
    """`ln N` at `theta` 0, at least `ln 2`, times `exp(theta)`."""
    similarity = Similarity()
    assert similarity.scale(torch.tensor(8)).item() == pytest.approx(math.log(8))
    assert similarity.scale(torch.tensor(1)).item() == pytest.approx(math.log(2))
    with torch.no_grad():
        similarity.theta.fill_(math.log(3))
    assert similarity.scale(torch.tensor(8)).item() == pytest.approx(3 * math.log(8))


def test_instance_term_of() -> None:
    made = term(0.5)
    assert made.weight == 0.5
    assert made.instance[0] == made.instance[2] == made.instance[4] != made.instance[1]
    assert made.subject.tolist() == [101, 101, 102, 102, 101, 0]
    with pytest.raises(ValueError, match="no training word"):
        InstanceTerm.of(1.0, WORDS.iloc[5:], CPU)


def test_positives() -> None:
    assert term().positives().subject.tolist() == [101, 101, 102, 102, 101, 0]


def test_head() -> None:
    assert isinstance(term().head(4), Similarity)


def split(n: int) -> Split:
    return Split.from_arrays(np.zeros((n, 4, 4), np.float32), np.arange(n) % 3, Spans.by_gap(np.arange(n) * 2.0), CPU)


def test_labels() -> None:
    assert torch.equal(term().labels(split(6)), torch.arange(6))
    with pytest.raises(ValueError, match="instance rows"):
        term().labels(split(5))


def test_pairs() -> None:
    """Positives are the same token from another subject; the same subject's other session is no candidate at all."""
    candidate, positive = term().pairs(TARGET.flatten())
    found = {(int(i), int(j)) for i, j in positive.nonzero()}
    assert found == {(0, 2), (2, 0), (1, 3), (3, 1), (2, 4), (4, 2)}
    assert not candidate[0, 4] and not candidate[4, 0]
    assert not candidate.diagonal().any()
    assert candidate[5, :6].sum() == 5
    assert not candidate[6:].any() and not candidate[:, 6:].any()


def reference(output: torch.Tensor, target: torch.Tensor) -> float:
    """The loss written out anchor by anchor."""
    flat, unit = target.flatten(), torch.nn.functional.normalize(output.flatten(0, 1), dim=-1)
    candidate, positive = term().pairs(flat)
    scale = math.log(int((flat != IGNORE).sum()))
    losses = []
    for i in range(len(flat)):
        if positive[i].any():
            logits = unit @ unit[i] * scale
            losses.append(logits[candidate[i]].logsumexp(0).item() - logits[positive[i]].mean().item())
    return sum(losses) / len(losses)


def test_unweighted() -> None:
    """The anchor-by-anchor InfoNCE; shared embeddings across subjects cost less; weight 0 adds exactly 0; a batch
    without a cross-subject pair costs 0 with finite gradients."""
    made, output = term(), embeddings()
    rows = torch.ones(TARGET.shape)
    assert made.unweighted(Reading(Similarity(), output, TARGET, rows)).item() == pytest.approx(
        reference(output, TARGET)
    )
    shared = output.clone()
    shared[0, 2], shared[1, 0] = shared[0, 0], shared[0, 1]
    assert made.unweighted(Reading(Similarity(), shared, TARGET, rows)) < made.unweighted(
        Reading(Similarity(), output, TARGET, rows)
    )
    off = term(0.0)
    assert off.loss(off.read(Similarity(), output, TARGET)).item() == 0.0
    lonely = torch.tensor([[0, 1, 5]])
    leaf = embeddings()[:1].requires_grad_()
    alone = made.unweighted(Reading(Similarity(), leaf, lonely, torch.ones(lonely.shape)))
    alone.backward()
    assert alone.item() == 0.0
    assert leaf.grad is not None
    assert torch.isfinite(leaf.grad).all()


SENTENCES = pd.DataFrame(
    {
        "run": ["sub-101_ses-0_task-MASC1_run-1"] * 3 + ["sub-102_ses-0_task-MASC1_run-1"] * 3,
        "position": [0, 1, 2, 0, 1, 2],
        "sentence": [0, 0, 1, 0, 0, 1],
    }
)
"""Two subjects hearing a two-word sentence (rows 0-1, 3-4) and the next sentence's first word (rows 2, 5)."""
HEARD = torch.tensor([[0, 1, 2], [3, 4, 5]])


def test_together_related() -> None:
    assert not Together().related(torch.tensor([0, 1, 1])).any()


def test_apart_of() -> None:
    assert Apart.of(SENTENCES, CPU).sentence.tolist() == [0, 0, 1, 0, 0, 1]


def test_apart_related() -> None:
    related = Apart(torch.tensor([0, 0, 1])).related(torch.tensor([0, 1, 2, 2]))
    assert related.tolist() == [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]]


def test_pairs_apart() -> None:
    """Apart leaves the anchor's sentence out of its negatives, whoever heard it, and keeps its positive; off keeps
    every other word a candidate."""
    together = InstanceTerm.of(1.0, SENTENCES, CPU)
    apart = InstanceTerm.of(1.0, SENTENCES, CPU, apart=True)
    assert isinstance(together.neighbours, Together) and not isinstance(together.neighbours, Apart)
    candidate, positive = apart.pairs(HEARD.flatten())
    assert candidate[0].nonzero().flatten().tolist() == [2, 3, 5]
    assert positive[0].nonzero().flatten().tolist() == [3]
    plain, plain_positive = together.pairs(HEARD.flatten())
    assert plain[0].nonzero().flatten().tolist() == [1, 2, 3, 4, 5]
    assert torch.equal(plain_positive, positive)


def chorus(tokens: int) -> pd.DataFrame:
    """Subjects 101 and 102 each hearing `tokens` one-word sentences of one story."""
    runs = ["sub-101_ses-0_task-MASC1_run-1"] * tokens + ["sub-102_ses-0_task-MASC1_run-1"] * tokens
    position = np.r_[np.arange(tokens), np.arange(tokens)]
    return pd.DataFrame({"run": runs, "position": position, "sentence": position})


def test_unweighted_shuffled_is_log_batch() -> None:
    """Unrelated embeddings cost about `ln` of the candidates; the same embedding across subjects costs far less."""
    tokens, width = 32, 1024
    made = InstanceTerm.of(1.0, chorus(tokens), CPU, apart=True)
    target = torch.arange(2 * tokens).view(2, tokens)
    rows = torch.ones(target.shape)
    shuffled = torch.randn(2, tokens, width, generator=torch.Generator().manual_seed(0))
    loss = made.unweighted(Reading(Similarity(), shuffled, target, rows)).item()
    assert loss == pytest.approx(math.log(2 * tokens - 1), abs=0.1)
    shared = shuffled[:1].expand(2, -1, -1)
    assert made.unweighted(Reading(Similarity(), shared, target, rows)).item() < loss - 1
