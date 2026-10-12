"""What simulated spans draw their beliefs from: the independent `refinerows.Bank` or `spancoupling.Correlated`."""

from typing import Protocol

import torch
from jaxtyping import Float, Int
from torch import Tensor


class Drawer(Protocol):
    """Draws one belief per position of spans of states, zeros past a span's end."""

    def draw(self, states: Int[Tensor, "b w"], generator: torch.Generator) -> Float[Tensor, "b w v"]: ...
