import torch

from megspeech.onsets import ContextConfig, OnsetEncoding

CONTEXT = ContextConfig(width=16, timescales=4)


def test_onset_encoding_forward() -> None:
    out = OnsetEncoding(CONTEXT).forward(torch.tensor([[0.0, 0.3], [0.0, 0.0]]))
    assert out.shape == (2, 2, CONTEXT.width)
    assert torch.equal(out[0, 0], out[1, 1])
