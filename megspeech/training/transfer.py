"""What a run's model starts from besides its initialisation: a trained run's whole weights, or its encoder's."""

from dataclasses import dataclass
from typing import ClassVar

import torch
from torch import Tensor

from core.paths import DataRoot

WEIGHTS = "model.pt"


@dataclass(frozen=True)
class TransferConfig:
    """The trained runs a model starts from and how fast what it took moves; the `optim.transfer` group, its leaves
    also under their flat `--optim.<name>` flags."""

    FLAT: ClassVar[bool] = True

    warm: str = ""
    """A trained run under models/ whose weights the model starts from instead of its initialisation, its context
    config the same as this run's: a run trained on every subject fine-tuned on the Deep subject alone. Empty starts
    from scratch."""
    stem: str = ""
    """A trained run under models/ whose encoder (scaler, whitener, subject layer, word CNN and word head) the model
    starts from, the span transformer and context head fresh; empty starts the encoder from scratch."""
    encoder_lr: float = 1.0
    """The encoder's learning rate as a share of `lr`, its decay alike; 0 freezes it: a `stem` encoder kept while the
    span transformer trains on top of it."""

    def weights(self, root: DataRoot, device: torch.device) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """The `warm` run's saved weights and the `stem` run's, under `root`'s models on `device`; none for an empty
        name, whose `root` is never read."""
        warm, stem = (
            torch.load(root.models() / name / WEIGHTS, map_location=device, weights_only=True)
            if name
            else dict[str, Tensor]()
            for name in (self.warm, self.stem)
        )
        return warm, stem
