import torch

from megspeech.data.labels.vocabulary import VOCAB_SIZE
from megspeech.evaluation.studies.refiner import FEATURES, STATES, Refiner


def test_forward() -> None:
    torch.manual_seed(0)
    inputs = torch.randn(2, 3, FEATURES)
    padding = torch.tensor([[False, False, False], [False, True, True]])
    logits = Refiner().forward(inputs, padding)
    assert logits.shape == (2, 3, STATES)
    assert torch.allclose(logits[..., :VOCAB_SIZE], inputs[..., :VOCAB_SIZE])
    assert torch.equal(logits[..., VOCAB_SIZE:], torch.zeros(2, 3, 1))
