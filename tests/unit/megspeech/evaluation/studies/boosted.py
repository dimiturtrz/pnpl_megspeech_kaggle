import numpy as np
import torch

from megspeech.data.labels.vocabulary import IGNORE
from megspeech.data.spans.spans import PAD
from megspeech.evaluation.studies.boosted import Booster, Boosting, Design, Rows, Side, Spanned

WORDS = 5
CLASSES = 3
VOCAB = 4


def spanned() -> Spanned:
    index = np.array([[0, 1, 2], [3, 4, PAD]])
    onset = np.array([[0.0, 0.5, 2.0], [0.0, 1.0, 0.0]])
    return Spanned(index, onset)


def rows(count: int = WORDS, offset: int = 0) -> Rows:
    generator = torch.Generator().manual_seed(0)
    near = torch.full((count, 2), PAD)
    near[1:, 0] = torch.arange(count - 1) + offset
    near[:-1, 1] = torch.arange(1, count) + offset
    return Rows(
        torch.randn(count, VOCAB, generator=generator).log_softmax(-1),
        torch.rand(count, CLASSES, generator=generator),
        torch.randn(count, VOCAB, generator=generator) - 20,
        torch.rand(count, 2, generator=generator) * 9,
        torch.arange(count) % VOCAB,
        near,
        torch.zeros(count, dtype=torch.long),
    )


def design() -> Design:
    return rows().design([Side.TAGS], neighbours=False)


def test_pairs() -> None:
    before, after, gap = spanned().pairs()
    assert before.tolist() == [0, 1, 3]
    assert after.tolist() == [1, 2, 4]
    assert gap.tolist() == [0.5, 1.5, 1.0]


def test_near() -> None:
    assert spanned().near(WORDS).tolist() == [[PAD, 1], [0, 2], [1, PAD], [PAD, 4], [3, PAD]]


def test_gaps() -> None:
    gaps = spanned().gaps(WORDS)
    assert gaps[1].tolist() == [0.5, 1.5]
    assert gaps[0].tolist() == [0.0, 0.5]
    assert gaps[4].tolist() == [1.0, 0.0]


def test_where() -> None:
    kept = design().where(torch.tensor([True, False, True, False, False]))
    assert len(kept.y) == 2
    assert kept.x.shape[0] == kept.base.shape[0] == 2


def test_labelled() -> None:
    marked = Design(design().x, design().base, torch.tensor([0, IGNORE, 1, IGNORE, 2]))
    assert marked.labelled().y.tolist() == [0, 1, 2]


def test_to() -> None:
    moved = design().to(torch.device("cpu"))
    assert torch.equal(moved.x, design().x)


def test_stacked() -> None:
    whole = Rows.stacked([rows(2), rows(3)])
    assert len(whole.y) == WORDS
    assert whole.near.tolist() == [[PAD, 1], [0, PAD], [PAD, 3], [2, 4], [3, PAD]]


def test_side() -> None:
    parts = rows()
    assert parts.side([]).shape == (WORDS, 0)
    both = parts.side([Side.TAGS, Side.TIMING])
    assert both.shape == (WORDS, CLASSES + 2)
    assert parts.side([Side.LEXICAL]).min() >= -12.0
    assert parts.side([Side.TIMING]).max() <= 3.0


def test_design() -> None:
    parts = rows()
    alone = parts.design([Side.TAGS], neighbours=False)
    assert alone.x.shape == (WORDS, VOCAB + CLASSES)
    around = parts.design([Side.TAGS], neighbours=True)
    assert around.x.shape == (WORDS, VOCAB + 3 * CLASSES + 2)
    assert torch.equal(around.x[1, VOCAB + CLASSES : VOCAB + 2 * CLASSES], parts.tags[0])
    assert torch.count_nonzero(around.x[0, VOCAB + CLASSES : VOCAB + 2 * CLASSES]) == 0
    assert around.x[0, -2:].tolist() == [0.0, 1.0]
    assert parts.design([], neighbours=True).x.shape == (WORDS, VOCAB)


def test_forward() -> None:
    data = design()
    model = Booster(data, hidden=8, dropout=0.0)
    assert torch.equal(model.forward(data.x, data.base), data.base)


def test_fit() -> None:
    data = design()
    model, curve = Boosting(seed=0, hidden=8, batch=2).fit(data, 3, data)
    assert len(curve) == 3
    assert model(data.x, data.base).shape == data.base.shape
    assert Boosting(seed=0, hidden=8, batch=2).fit(data, 1)[1] == []


def test_loss() -> None:
    data = design()
    model = Booster(data, hidden=8, dropout=0.0)
    expected = torch.nn.functional.cross_entropy(data.base, data.y).item()
    assert abs(Boosting.loss(model, data) - expected) < 1e-6


def test_predict() -> None:
    data = design()
    predicted = Boosting.predict(Booster(data, hidden=8, dropout=0.0), data)
    assert torch.allclose(predicted.exp().sum(-1), torch.ones(WORDS), atol=1e-5)
