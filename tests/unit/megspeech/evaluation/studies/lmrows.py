import torch

from megspeech.data.labels.vocabulary import IGNORE, VOCAB_SIZE
from megspeech.data.spans.spans import PAD
from megspeech.evaluation.scoring.score import Head
from megspeech.evaluation.studies.lmrows import Posteriors


class Split:
    y = torch.tensor([1, IGNORE, 2])
    index = torch.tensor([[0, 1], [2, PAD]])


def posteriors() -> Posteriors:
    logits = {Head.CONTEXT: torch.zeros(3, VOCAB_SIZE, dtype=torch.float16), Head.WORD: torch.ones(3, VOCAB_SIZE)}
    return Posteriors.of(Split(), logits)  # pyrefly: ignore[bad-argument-type]


def test_of() -> None:
    rows = posteriors()
    assert rows.head.dtype == torch.float32
    assert float(rows.word[0, 0]) == 1.0
    assert torch.equal(rows.labels, Split.y)


def test_record() -> None:
    assert set(posteriors().record()) == {"head", "word", "labels", "index"}


def test_restored() -> None:
    rows = posteriors()
    back = Posteriors.restored(rows.record())
    assert torch.equal(back.index, rows.index)
    assert torch.equal(back.head, rows.head)
