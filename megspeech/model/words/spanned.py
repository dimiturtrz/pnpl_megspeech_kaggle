"""MEG-XL fine-tuned end to end over each span's whole continuous segment, one word's feature read at its onset.

MEG-XL was pretrained on 150 s of continuous MEG, not on 1 s windows: its temporal attention reads the words around a
word. One pass of the trunk over a span's segment (`Split.segmented`) gives every channel's tokens, each 240 ms; a word
is read from the tokens covering its first `read` samples, each token reduced by a learned projection shared over
channels, the channels' components flattened into the word's embedding. The trunk trains at a fraction of the other
layers' rate (`TransferConfig.encoder_lr`), so the pretrained weights move gently.
"""

from dataclasses import dataclass
from typing import override

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn

from core.paths import DataRoot
from megspeech.data.constants import RATE
from megspeech.model.words.base import WordEncoder
from megspeech.model.words.conv import ConvConfig
from megspeech.model.words.trunk import DOWNSAMPLING, LATENT, Trunk

WORD = RATE // 2
"""Samples after its onset whose tokens make a word's feature: MEG-XL's own word read."""


@dataclass(frozen=True)
class SpannedConfig:
    components: int = 16
    """Features each channel's token is projected to, then flattened over the channels: 306 x 16 as in the frozen
    spanned read's whitened components, here learned."""
    width: int = 512
    """The word embedding's width, after one linear layer over the flattened components."""
    read: int = WORD
    """Samples after a word's onset whose tokens are averaged into its feature."""
    full_precision: bool = False
    """Run the trunk in float32 rather than under the caller's bfloat16 autocast: a stabler fine-tune at about twice the
    activation memory."""


SPANNED = SpannedConfig()


class SpanWords(WordEncoder):
    """Words read out of one trunk pass over their span's segment, so a batch's rows are spans, not words."""

    def __init__(self, trunk: Trunk, conv: ConvConfig, spanned: SpannedConfig, channels: int) -> None:
        super().__init__(spanned.width, instance=conv.instance)
        self.trunk = trunk
        self.read = spanned.read
        self.full = spanned.full_precision
        self.reduce = nn.Sequential(nn.LayerNorm(LATENT), nn.Linear(LATENT, spanned.components))
        flat = channels * spanned.components
        self.mix = nn.Sequential(nn.Dropout(conv.dropout), nn.Linear(flat, spanned.width), nn.GELU())
        self.head = self.classifier(conv.dropout, conv.head)

    @classmethod
    def pretrained(cls, root: DataRoot, conv: ConvConfig, spanned: SpannedConfig, channels: int) -> "SpanWords":
        return cls(Trunk.pretrained(root), conv, spanned, channels)

    @override
    def backbone(self) -> tuple[nn.Parameter, ...]:
        return tuple(self.trunk.parameters())

    @override
    def positions(
        self, x: Float[Tensor, "g c t"], onset: Float[Tensor, "g w"], slot: Int[Tensor, "g w"]
    ) -> Float[Tensor, "g w e"]:
        """Each span's words' embeddings, zeros at empty positions (`slot == len(x)`)."""
        mask = slot < len(x)
        x = nn.functional.pad(x, (0, -x.shape[-1] % DOWNSAMPLING))
        with torch.autocast(x.device.type, enabled=not self.full):
            tokens = self.trunk(x.float() if self.full else x)
        weights = self.weights((onset * RATE).round().long(), tokens.shape[2], self.read) * mask[..., None]
        pooled = torch.einsum("gckf,gwk->gwcf", self.reduce(tokens), weights.to(tokens.dtype))
        return self.mix(pooled.flatten(2)) * mask[..., None]

    @override
    def embed(self, x: Float[Tensor, "b c t"], valid: Bool[Tensor, " b"] | None = None) -> Float[Tensor, "b e"]:
        """A window as a span of one word at its first sample."""
        rows = torch.arange(len(x), device=x.device)[:, None]
        return self.positions(x, x.new_zeros(len(x), 1), rows)[:, 0]

    @staticmethod
    def weights(start: Int[Tensor, "g w"], tokens: int, read: int = WORD) -> Float[Tensor, "g w k"]:
        """Equal weights on the tokens overlapping each word's `read` samples from sample `start` of its segment."""
        token = torch.arange(tokens, device=start.device)
        covered = (token >= start[..., None] // DOWNSAMPLING) & (token <= (start[..., None] + read - 1) // DOWNSAMPLING)
        return covered.float() / covered.sum(-1, keepdim=True).clamp(min=1)
