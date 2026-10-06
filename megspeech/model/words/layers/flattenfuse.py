"""The fixed-layout fuse of a channel-first front end."""

from typing import override

from jaxtyping import Bool, Float
from torch import Tensor, nn


class FlattenFuse(nn.Conv1d):
    """The fixed-layout fuse: all `sites * out` features through one 1x1 convolution; dropped sites are zeroed and
    the survivors rescaled by 1 / (1 - p) when dropout is active."""

    def __init__(self, sites: int, out: int, hidden: int, p: float = 0.0) -> None:
        super().__init__(sites * out, hidden, 1)
        self.scale = 1.0 / (1.0 - p)

    @override
    def forward(
        self, input: Float[Tensor, "b s d t"], keep: Bool[Tensor, "b s"] | None = None
    ) -> Float[Tensor, "b h t"]:
        if keep is not None:
            input = input * keep[:, :, None, None] * self.scale
        return super().forward(input.flatten(1, 2))
