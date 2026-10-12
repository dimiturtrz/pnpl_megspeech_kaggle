import torch

from megspeech.data.labels.vocabulary import VOCAB_SIZE
from megspeech.evaluation.studies.drawer import Drawer
from megspeech.evaluation.studies.refinerows import Bank
from tests.unit.megspeech.evaluation.studies.spancoupling import every_state


def test_draw() -> None:
    drawer: Drawer = Bank.spanned(every_state())
    drawn = drawer.draw(torch.tensor([[0, 1]]), torch.Generator().manual_seed(0))
    assert drawn.shape == (1, 2, VOCAB_SIZE)
