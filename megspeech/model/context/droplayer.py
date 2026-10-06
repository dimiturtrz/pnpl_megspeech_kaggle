"""A pre-norm transformer layer whose residual branches are stochastic-depth droppable."""

from typing import override

from jaxtyping import Bool, Float
from torch import Tensor, nn

from megspeech.model.context.droppath import DropPath


class DropPathLayer(nn.TransformerEncoderLayer):
    """A pre-norm transformer layer whose attention and feed-forward branches each pass through `path` in training;
    with rate 0 or in evaluation exactly `nn.TransformerEncoderLayer`."""

    def __init__(self, width: int, heads: int, feedforward: int, dropout: float) -> None:
        super().__init__(width, heads, feedforward, dropout, batch_first=True, norm_first=True)
        self.path = DropPath()

    @override
    def forward(
        self,
        src: Float[Tensor, "b s d"],
        src_mask: Float[Tensor, "s s"] | None = None,
        src_key_padding_mask: Bool[Tensor, "b s"] | None = None,
        is_causal: bool = False,
    ) -> Float[Tensor, "b s d"]:
        if not self.training or not self.path.rate:
            return super().forward(src, src_mask, src_key_padding_mask, is_causal)
        x = src + self.path(self._sa_block(self.norm1(src), src_mask, src_key_padding_mask, is_causal))
        return x + self.path(self._ff_block(self.norm2(x)))
