import torch
from torch import nn

from megspeech.model import SentenceNet
from megspeech.onsets import ContextConfig
from megspeech.wordnet import CHANNELS, WORDS, ConvConfig

SMALL = ConvConfig(hidden=8, dilations=(1, 2))
CONTEXT = ContextConfig(width=16, heads=2, layers=1)
SAMPLES = 50
GENERATOR = torch.Generator().manual_seed(0)


def test_sentence_net_forward() -> None:
    model = SentenceNet(nn.Identity(), SMALL, CONTEXT).eval()
    mask = torch.tensor([[True, True, False], [True, False, False]])
    x = torch.randn(3, CHANNELS, SAMPLES, generator=GENERATOR)
    context, word = model.forward(x, torch.tensor([[0.0, 0.2, 0.0], [0.0, 0.0, 0.0]]), mask)
    assert context.shape == word.shape == (2, 3, WORDS)
    assert torch.isfinite(context[mask]).all()
