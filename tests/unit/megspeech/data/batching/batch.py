import torch

from megspeech.data.batching.batch import Batch
from megspeech.data.labels.vocabulary import IGNORE
from megspeech.data.spans.spans import PAD


def test_target() -> None:
    slot = torch.tensor([[0, 1, 2]])
    batch = Batch(torch.zeros(2, 1, 1), torch.zeros(1, 3), slot, torch.tensor([[4, 3, PAD]]), torch.zeros(2))
    assert batch.target(torch.tensor([0, 1, 2, 7, 8])).tolist() == [[8, 7, IGNORE]]
