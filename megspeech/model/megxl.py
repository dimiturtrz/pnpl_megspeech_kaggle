"""MEG-XL's pretrained criss-cross trunk (Jayalath et al. 2026, MIT), loaded from its pinned checkout.

The trunk tokenises each channel with its frozen BioCodec (12x downsampling, so one token per 240 ms at 50 Hz),
embeds every token with its sensor's position, orientation and type, and alternates temporal attention along each
channel with spatial attention across the channels. Its code and its weights are fetched by
`python -m megspeech.data.fetch --weights`; its imports beyond torch come with the `megxl` extra.
"""

import importlib
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import override

import numpy as np
import torch
from jaxtyping import Float, Int
from torch import Tensor, nn

from core.paths import DataRoot
from megspeech.data.fetch import MEGXL, NEUROMAG

CHECKPOINT = "meg-xl-med-v2.ckpt"
"""The released medium model, the one whose word-decoding numbers the paper reports."""

MAGNETOMETER = 3024
"""The Neuromag coil type of a magnetometer; its other channels are planar gradiometers."""

DOWNSAMPLING = 12
"""Samples per BioCodec token: its encoder strides 3 x 2 x 2."""

ROTATIONS = "rope_embedding_layer.rotate"
"""The checkpoint's RoPE tables, sized to its pretraining length; the trunk recomputes them for any length."""


@dataclass(frozen=True)
class Geometry:
    """Every channel's normalised position, coil orientation and type (1 magnetometer, 0 gradiometer), in the
    recordings' channel order: their names ascending."""

    xyz: Float[np.ndarray, "c 3"]
    abc: Float[np.ndarray, "c 3"]
    kind: Int[np.ndarray, " c"]

    @classmethod
    def of(cls, channels: Sequence[Mapping[str, object]]) -> "Geometry":
        """From MNE channel records: position `loc[0:3]`, orientation the coil normal `loc[9:12]` of a
        magnetometer or the gradient axis `loc[3:6]` of a gradiometer, as MEG-XL read them; positions centred and
        scaled to unit root-mean-square radius over 3 axes."""
        meg = sorted((c for c in channels if str(c["ch_name"]).startswith("MEG")), key=lambda c: str(c["ch_name"]))
        loc = np.stack([np.asarray(c["loc"], dtype=np.float64) for c in meg])
        kind = np.array([c["coil_type"] == MAGNETOMETER for c in meg], dtype=np.int64)
        xyz = loc[:, :3] - loc[:, :3].mean(axis=0)
        xyz /= np.sqrt(3 * np.mean(np.sum(xyz**2, axis=1)))
        abc = np.where(kind[:, None] == 1, loc[:, 9:12], loc[:, 3:6])
        return cls(xyz, abc, kind)

    @classmethod
    def neuromag(cls, path: Path) -> "Geometry":
        """From the channel records of a Neuromag-306 recording at `path`."""
        info = importlib.import_module("mne.io").read_info(path, verbose="error")
        return cls.of(info["chs"])


@dataclass(frozen=True)
class Checkpoint:
    """Where the trunk's code and weights are."""

    weights: Path
    code: Path = MEGXL.home()

    @classmethod
    def under(cls, root: DataRoot) -> "Checkpoint":
        return cls(root.foreign(MEGXL.repo.split("/")[-1]) / CHECKPOINT)

    def modules(self) -> tuple[ModuleType, ModuleType]:
        """The checkout's tokenizer and trunk modules, importable once the checkout is on the path."""
        if str(self.code) not in sys.path:
            sys.path.insert(0, str(self.code))
        return (
            importlib.import_module("brainstorm.neuro_tokenizers.biocodec.model"),
            importlib.import_module("brainstorm.models.criss_cross_transformer"),
        )

    def trunk(self) -> nn.Module:
        """The pretrained trunk, its tokenizer's weights among the checkpoint's."""
        biocodec, crisscross = self.modules()
        saved = torch.load(self.weights, map_location="cpu", weights_only=False)
        hparams = {key: value for key, value in saved["hyper_parameters"].items() if key != "debug_nan_checks"}
        tokenizer = biocodec.BioCodecModel._get_optimized_model()  # noqa: SLF001
        module = crisscross.CrissCrossTransformerModule(tokenizer=tokenizer, **hparams)
        state = {key: value for key, value in saved["state_dict"].items() if ROTATIONS not in key}
        missing, unexpected = module.load_state_dict(state, strict=False)
        if unexpected or any(ROTATIONS not in key for key in missing):
            raise ValueError(f"{self.weights.name}: missing {missing}, unexpected {unexpected}")
        return module


class Trunk(nn.Module):
    """MEG-XL's features of every channel's tokens, its sensors fixed to one geometry."""

    xyz: Tensor
    abc: Tensor
    kind: Tensor

    def __init__(self, module: nn.Module, geometry: Geometry) -> None:
        super().__init__()
        self.module = module
        self.register_buffer("xyz", torch.as_tensor(geometry.xyz, dtype=torch.float32))
        self.register_buffer("abc", torch.as_tensor(geometry.abc, dtype=torch.float32))
        self.register_buffer("kind", torch.as_tensor(geometry.kind))

    @classmethod
    def pretrained(cls, root: DataRoot) -> "Trunk":
        return cls(Checkpoint.under(root).trunk(), Geometry.neuromag(NEUROMAG.home(root)))

    @override
    def forward(self, x: Float[Tensor, "b c t"]) -> Float[Tensor, "b c k d"]:
        """Features of `x`'s `t // DOWNSAMPLING` tokens per channel."""
        batch = len(x)
        return self.module(
            x,
            self.xyz.expand(batch, -1, -1),
            self.abc.expand(batch, -1, -1),
            self.kind.expand(batch, -1),
            apply_mask=False,
        )["features"]
