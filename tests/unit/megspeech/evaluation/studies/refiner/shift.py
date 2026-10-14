import torch

from megspeech.data.labels.vocabulary import VOCAB_SIZE
from megspeech.data.spans.spans import PAD
from megspeech.evaluation.studies.refiner.shift import Offsets, Shifted
from megspeech.evaluation.studies.refiner.wiki import OOV
from tests.unit.megspeech.evaluation.studies.refiner.coupling import every_state


def test_offsets_of() -> None:
    keys = torch.tensor([0, 0, 1, 1])
    beliefs = torch.tensor([[0.0], [0.0], [2.0], [2.0]])
    made = Offsets.of(keys, torch.zeros(4, dtype=torch.long), beliefs)
    assert made.keys.tolist() == [0, 1]
    assert made.offsets.flatten().tolist() == [-1.0, 1.0]


def test_offsets_of_own_level_is_zero() -> None:
    keys = torch.tensor([0, 1])
    assert Offsets.of(keys, keys, torch.randn(2, 3)).offsets.abs().max() == 0


def test_at() -> None:
    made = Offsets(torch.tensor([2, 5]), torch.tensor([[1.0], [3.0]]))
    assert made.at(torch.tensor([[5, 4, 2]])).flatten().tolist() == [3.0, 0.0, 1.0]


def test_shifted_of() -> None:
    made = Shifted.of(every_state())
    assert len(made.levels) == 2
    assert len(made.rows.levels) == 1


def test_draw() -> None:
    states = torch.tensor([[3, 4, OOV], [7, PAD, PAD]])
    drawn = Shifted.of(every_state()).draw(states, torch.Generator().manual_seed(0))
    assert drawn.shape == (2, 3, VOCAB_SIZE)
    assert bool((drawn[1, 1:] == 0).all())
    assert torch.allclose(drawn[0].exp().sum(-1), torch.ones(3))
