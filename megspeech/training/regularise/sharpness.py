"""Sharpness-aware minimisation (Foret et al. 2021, arXiv 2010.01412): each step descends along the gradient taken at
the worst point of a `rho`-ball around the weights, so training settles in minima flat enough that the shift from
fitted to unseen MEG moves the loss less. The encoder memorises its training MEG (fit-minus-val MEG excess .125 @1,
megspeech-d6r); a flat minimum is the generalisation mechanism no regulariser here has tried."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import override

import torch
from jaxtyping import Float
from torch import Tensor, nn

EPS = 1e-12
"""Keeps the ascent finite when every gradient is zero."""


class Flat:
    """Plain descent: the gradient at the weights themselves."""

    def backward(
        self, value: Float[Tensor, ""], again: Callable[[], Float[Tensor, ""]], params: tuple[nn.Parameter, ...]
    ) -> None:
        """Leaves on `params` the gradient of `value`; `again` recomputes it at the current weights."""
        del again, params
        value.backward()


@dataclass(frozen=True)
class Sharp(Flat):
    """SAM: after `value`'s gradient, the weights step `rho` along it (normalised over all `params`), `again`'s gradient
    there replaces it, and the weights step back."""

    rho: float

    @classmethod
    def of(cls, rho: float) -> Flat:
        """Sharpness-aware at radius `rho`; plain descent when it is 0."""
        return cls(rho) if rho else Flat()

    @override
    def backward(
        self, value: Float[Tensor, ""], again: Callable[[], Float[Tensor, ""]], params: tuple[nn.Parameter, ...]
    ) -> None:
        value.backward()
        graded = [(p, p.grad) for p in params if p.grad is not None]
        norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(grad) for _, grad in graded]))
        scale = self.rho / (norm + EPS)
        with torch.no_grad():
            ascent = [(p, grad * scale) for p, grad in graded]
            for p, step in ascent:
                p.add_(step)
                p.grad = None
        again().backward()
        with torch.no_grad():
            for p, step in ascent:
                p.sub_(step)
