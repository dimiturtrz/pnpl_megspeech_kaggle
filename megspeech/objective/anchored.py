"""The anchored context head: scaled cosine to fixed per-word anchors (`objective.anchors`)."""

import math
from typing import override

import torch
from jaxtyping import Float
from torch import Tensor, nn
from torch.nn import functional

INITIAL_SCALE = 10.0
"""The head's initial logit scale."""


class AnchoredHead(nn.Module):
    """Logits as `exp(scale)` times the cosine between a linear projection of the head input and each word's anchor:
    frozen buffers, or from-init parameters when `trainable`."""

    anchors: Tensor

    def __init__(self, width: int, anchors: Float[Tensor, "v d"], *, trainable: bool = False) -> None:
        super().__init__()
        self.project = nn.Linear(width, anchors.shape[1])
        self.scale = nn.Parameter(torch.tensor(math.log(INITIAL_SCALE)))
        if trainable:
            self.anchors = nn.Parameter(anchors.float().clone())
        else:
            self.register_buffer("anchors", anchors.float().clone())

    @override
    def forward(self, h: Float[Tensor, "*b w"]) -> Float[Tensor, "*b v"]:
        unit = functional.normalize(self.project(h).float(), dim=-1)
        return self.scale.exp() * (unit @ functional.normalize(self.anchors, dim=-1).T)
