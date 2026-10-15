"""The compute devices the project names: one vocabulary for torch devices, autocast and `map_location`."""

from dataclasses import dataclass
from enum import StrEnum

import torch

from core.paths import DataRoot


class Device(StrEnum):
    CUDA = "cuda"
    CPU = "cpu"

    @classmethod
    def best(cls) -> "Device":
        """The GPU when one is visible, else the CPU."""
        return cls.CUDA if torch.cuda.is_available() else cls.CPU

    def as_torch(self) -> torch.device:
        return torch.device(self.value)

    @classmethod
    def autocast(cls) -> torch.autocast:
        """bfloat16 autocast on the GPU; full precision, silently, when no GPU is visible."""
        return torch.autocast(cls.CUDA, dtype=torch.bfloat16, enabled=torch.cuda.is_available())


@dataclass(frozen=True)
class Placed:
    """A model or LM named under the data root, and the device it is read onto."""

    root: DataRoot
    name: str
    device: torch.device
