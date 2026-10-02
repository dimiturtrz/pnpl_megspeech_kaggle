"""Low-rank adapters (Hu et al. 2021): a frozen linear layer plus a trainable update `up @ down`, zero at start."""

from typing import override

from jaxtyping import Float
from torch import Tensor, nn

TARGETS = frozenset({"qkv", "proj"})
"""MEG-XL's attention projections, `SelfAttention.qkv` and `.proj`, in both its temporal and spatial blocks."""


class Lora(nn.Module):
    def __init__(self, base: nn.Linear, rank: int) -> None:
        super().__init__()
        self.base = base.requires_grad_(requires_grad=False)
        self.down = nn.Linear(base.in_features, rank, bias=False)
        self.up = nn.Linear(rank, base.out_features, bias=False)
        nn.init.zeros_(self.up.weight)

    @override
    def forward(self, x: Float[Tensor, "*b i"]) -> Float[Tensor, "*b o"]:
        return self.base(x) + self.up(self.down(x))

    def adapter(self) -> list[nn.Parameter]:
        return [*self.down.parameters(), *self.up.parameters()]

    @classmethod
    def inject(cls, module: nn.Module, rank: int, targets: frozenset[str] = TARGETS) -> int:
        """Wrap every linear child of `module` named in `targets` at `rank`, none at rank 0; how many it wrapped."""
        found = [
            (parent, name, child)
            for parent in module.modules()
            for name, child in parent.named_children()
            if rank and name in targets and isinstance(child, nn.Linear)
        ]
        for parent, name, child in found:
            setattr(parent, name, cls(child, rank))
        return len(found)
