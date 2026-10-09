import pandas as pd
import pytest
import torch

from megspeech.data.labels.vocabulary import VOCAB_SIZE
from megspeech.data.spans.spans import PAD
from megspeech.evaluation.studies.crowding import Rows, Section
from megspeech.evaluation.studies.lmbound import (
    LAMS,
    RIGHT,
    Backoff,
    Bound,
    Causal,
    Context,
    Drop,
    Mask,
    Spelling,
    Study,
    Table,
)

WORDS = ["Sherlock", "said", "that", "it", "was", "the", "dog"]
"""`said`, `Sherlock` and `dog` are out of the vocabulary; the rest are in it."""
AT = 4
WIDTH = RIGHT + 1


def score(key: Context) -> torch.Tensor:
    """A fake LM: candidate `len(left) % 50` is likely, whatever the right words."""
    out = torch.zeros(VOCAB_SIZE, WIDTH)
    out[len(key[0]) % VOCAB_SIZE] = 5.0
    return out


def test_mask_shown() -> None:
    assert Mask().shown(WORDS, AT) == (("Sherlock", "said", "that", "it"), ("the", "dog"))


def test_known() -> None:
    assert Mask.known("The")
    assert not Mask.known("Sherlock")


def test_backoff_shown() -> None:
    assert Backoff().shown(WORDS, AT) == (("that", "it"), ("the",))


def test_drop_shown() -> None:
    assert Drop().shown(WORDS, AT) == (("that", "it"), ("the",))
    assert Drop().shown(["it", "Sherlock", "was", "the"], 2) == (("it",), ("the",))


def test_spelling_of() -> None:
    spelling = Spelling.of(pd.Series(["The", "the", "the", "dog"]))
    assert sorted(spelling.forms.values()) == ["dog", "the"]


class Spans:
    token = torch.tensor([1, 2, 3])
    index = torch.tensor([[0, 1], [2, PAD]])


def test_spans() -> None:
    spelling = Spelling({1: "the", 2: "dog", 3: "ran"})
    out = spelling.spans(Spans(), torch.tensor([1, 2]))  # pyrefly: ignore[bad-argument-type]
    assert out == [(["The", "dog"], 1), (["Ran"], 0)]


def test_pieces() -> None:
    lm = Causal(torch.nn.Module(), lambda texts: [[len(text)] for text in texts], 0)  # pyrefly: ignore[bad-argument-type]
    assert lm.pieces(["a", "bc"]) == [[2], [3]]
    assert lm.pieces([]) == []


def test_cumulative() -> None:
    first = torch.log(torch.tensor([0.5, 0.25, 0.25]))
    after = torch.log(torch.full((1, 2, 3), 1 / 3))
    out = Causal.cumulative(first, after, torch.tensor([[0, 1]]), torch.tensor([[1, 2]]))
    assert out[0, 0] == pytest.approx(float(first[0]))
    assert out[0, 1] == pytest.approx(float(first[0]) + float(torch.log(torch.tensor(1 / 3))))


def test_scored() -> None:
    table = Table.scored(score, [(("a",), ()), (("a",), ()), ((), ())])
    assert len(table.scores) == 2
    assert int(table.unigram.argmax()) == 0


def test_pmi() -> None:
    table = Table.scored(score, [(("a",), ())])
    pmi = table.pmi([(("a",), ())], 0)
    assert pmi.shape == (1, VOCAB_SIZE)
    assert int(pmi[0].argmax()) == 1
    assert float(pmi[0, 0]) < 0


def section() -> Section:
    labels = torch.tensor([1, 0])
    head = torch.zeros(2, VOCAB_SIZE).log_softmax(-1)
    return Section({"a": Rows(head, head, labels)}, {}, head[0])


def bound() -> Bound:
    return Bound.of(section(), {"a": [(WORDS, AT), (WORDS, 0)]})


def test_bound_of() -> None:
    assert bound().contexts["backoff"]["a"] == [(("that", "it"), ("the",)), ((), ())]


def test_wanted() -> None:
    assert len(bound().wanted()) == 6


def test_masks() -> None:
    assert bound().masks() == ("full", "backoff", "drop")


def test_fused() -> None:
    table = Table.scored(score, bound().wanted())
    fused = bound().fused(table, "full", 0)
    assert int(fused["a"].other[0].argmax()) == 4
    assert torch.equal(fused["a"].labels, torch.tensor([1, 0]))


def test_run() -> None:
    labels = torch.arange(64) % VOCAB_SIZE
    head = torch.zeros(64, VOCAB_SIZE).log_softmax(-1)
    words = [(["it", "was", "the", "dog"], int(i) % 4) for i in range(64)]
    graded = Bound.of(Section({"a": Rows(head, head, labels)}, {}, head[0]), {"a": words})
    table = Table.scored(score, graded.wanted())
    result = Study(graded, graded).run(table, 0)
    assert set(result) == {
        f"{mask}/{channel}" for mask in ("full", "backoff", "drop") for channel in ("pmi", "pmi-right")
    }
    assert len(LAMS) > 1
