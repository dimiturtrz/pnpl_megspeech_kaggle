"""The sentence context's Conformer: each transformer layer followed by a depthwise convolution across neighbouring
words (Gulati et al. 2020's convolution module), so local word order is mixed without attention having to learn it."""

from typing import cast, override

from jaxtyping import Bool, Float
from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from megspeech.model.context.droplayer import DropPathLayer
from megspeech.model.context.droppath import DropPath
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
        self.paths = nn.ModuleList(DropPath() for _ in range(layers))

    def ramp(self, peak: float) -> None:
        """Stochastic depth on every layer's branches, the rate rising linearly to `peak` at the last layer."""
        DropPath.ramp(self.paths, peak)
        DropPath.ramp((attend.path for attend in self.attend if isinstance(attend, DropPathLayer)), peak)

    @staticmethod
    def layer(width: int, heads: int, dropout: float) -> DropPathLayer:
        """One pre-norm transformer layer over a span's words, its residual branches droppable."""
        return DropPathLayer(width, heads, EXPANSION * width, dropout)

    @override
    def forward(self, h: Float[Tensor, "g w d"], src_key_padding_mask: Bool[Tensor, "g w"]) -> Float[Tensor, "g w d"]:
        with sdpa_kernel(SDPBackend.MATH):
            for attend, mix, path in zip(self.attend, self.mix, self.paths, strict=True):
                attended = attend(h, src_key_padding_mask=src_key_padding_mask)
                mixed = mix(attended, src_key_padding_mask)
                h = cast(DropPath, path).residual(attended, mixed)
        return h
