"""The sentence context's Conformer: each transformer layer followed by a depthwise convolution across neighbouring
words (Gulati et al. 2020's convolution module), so local word order is mixed without attention having to learn it."""

from typing import override

from jaxtyping import Bool, Float
from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from megspeech.model.context.mixing import Mixing
from megspeech.model.words.layers.conformer import EXPANSION


class SpanConformer(nn.Module):
    """`nn.TransformerEncoder`'s call over a span of words, a word-wise convolution module after every layer.

    Attention runs on the math kernel: under `torch.compile` the memory-efficient kernel's backward rejects the
    padding mask's layout ("(*bias): last dimension must be contiguous"), and spans are too short for it to matter.
    """

    def __init__(self, width: int, heads: int, layers: int, dropout: float, kernel: int) -> None:
        super().__init__()
        self.attend = nn.ModuleList(self.layer(width, heads, dropout) for _ in range(layers))
        self.mix = nn.ModuleList(Mixing(width, dropout, kernel) for _ in range(layers))

    @staticmethod
    def layer(width: int, heads: int, dropout: float) -> nn.TransformerEncoderLayer:
        """One pre-norm transformer layer over a span's words."""
        return nn.TransformerEncoderLayer(width, heads, EXPANSION * width, dropout, batch_first=True, norm_first=True)

    @override
    def forward(self, h: Float[Tensor, "g w d"], src_key_padding_mask: Bool[Tensor, "g w"]) -> Float[Tensor, "g w d"]:
        with sdpa_kernel(SDPBackend.MATH):
            for attend, mix in zip(self.attend, self.mix, strict=True):
                h = mix(attend(h, src_key_padding_mask=src_key_padding_mask), src_key_padding_mask)
        return h
