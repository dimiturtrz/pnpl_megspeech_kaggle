"""How an encoder moves while the model around it trains: with it, or held as it is."""

from typing import override

from torch import nn


class Moving:
    """An encoder that trains with the rest of the model: nothing of it is held."""

    def hold(self, stage: tuple[nn.Module, ...]) -> None:
        """Nothing is held."""


class Frozen(Moving):
    """An encoder kept as it was (`TransferConfig.encoder_lr` 0): no gradient reaches or flows through it, and its
    dropout and batch statistics stay in evaluation mode however the model's mode changes."""

    @override
    def hold(self, stage: tuple[nn.Module, ...]) -> None:
        for module in stage:
            module.train(mode=False)
            module.requires_grad_(requires_grad=False)
