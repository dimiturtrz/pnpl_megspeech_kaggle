import torch
from torch import nn

from megspeech.data.batching.split import Split
from megspeech.data.batching.windows import Windows
from megspeech.data.constants import CHANNELS, RATE
from megspeech.features.trained import Trained
from megspeech.model.sentence import SentenceNet
from tests.unit.megspeech.model.sentence import CONTEXT

WORDS = 5


def split() -> Split:
    meg = torch.randn(CHANNELS, WORDS * RATE + RATE, generator=torch.Generator().manual_seed(0))
    at = torch.arange(WORDS) * RATE
    zeros = torch.zeros(WORDS, dtype=torch.long)
    index = torch.arange(WORDS)[None]
    return Split(Windows(meg, at, at + RATE), zeros, zeros, index, index.float(), torch.ones(WORDS), zeros)


def test_call() -> None:
    net = SentenceNet(nn.Identity(), CONTEXT).eval()
    words = split()
    whole = Trained(net)(words)
    assert whole.shape == (WORDS, CONTEXT.conv.width, 1)
    assert whole.dtype == torch.bfloat16
    assert torch.allclose(Trained(net, chunk=2)(words).float(), whole.float(), atol=1e-2)
    with torch.no_grad():
        exact = net.embed(words.windows(torch.arange(WORDS)), words.subject)
    assert torch.allclose(whole[..., 0].float(), exact, atol=0.05, rtol=0.05)
