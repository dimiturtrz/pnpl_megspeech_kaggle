"""The compute devices the project names: one vocabulary for torch devices, autocast and `map_location`."""

from enum import StrEnum

import torch


class Device(StrEnum):
    CUDA = "cuda"
    CPU = "cpu"

    @classmethod
    def best(cls) -> "Device":
        """The GPU when one is visible, else the CPU."""
        return cls.CUDA if torch.cuda.is_available() else cls.CPU

    def as_torch(self) -> torch.device:
        return torch.device(self.value)
