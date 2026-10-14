"""The span refiner `lmrefine` trains: a small transformer over a span's per-row beliefs, residual on each row's own
log-probabilities so that, untrained, it returns the base."""

from typing import override

import torch
from jaxtyping import Bool, Float
from torch import Tensor, nn

from megspeech.data.labels.vocabulary import VOCAB_SIZE
from megspeech.data.spans.spans import MAX_WORDS

STATES = VOCAB_SIZE + 1
"""The 50 words, then `OOV`."""
FEATURES = VOCAB_SIZE + 2
"""A row's 50 log-probabilities, then whether it shows `OOV` and whether it shows its true state at all."""
WIDTH = 128
HEADS = 4
LAYERS = 3
DROPOUT = 0.1


class Refiner(nn.Module):
    """A span transformer over beliefs: logits over the 50 (residual on the row's own) and `OOV`."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Linear(FEATURES, WIDTH)
        self.slot = nn.Embedding(MAX_WORDS, WIDTH)
        self.back = nn.Embedding(MAX_WORDS, WIDTH)
        layer = nn.TransformerEncoderLayer(WIDTH, HEADS, 2 * WIDTH, DROPOUT, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, LAYERS, enable_nested_tensor=False)
        self.out = nn.Linear(WIDTH, STATES)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    @override
    def forward(self, inputs: Float[Tensor, "b w f"], padding: Bool[Tensor, "b w"]) -> Float[Tensor, "b w s"]:
        slot = torch.arange(inputs.shape[1], device=inputs.device)
        back = ((~padding).sum(1, keepdim=True) - 1 - slot).clamp(min=0)
        hidden = self.encoder(self.embed(inputs) + self.slot(slot) + self.back(back), src_key_padding_mask=padding)
        out = self.out(hidden)
        return torch.cat([out[..., :VOCAB_SIZE] + inputs[..., :VOCAB_SIZE], out[..., VOCAB_SIZE:]], dim=-1)
