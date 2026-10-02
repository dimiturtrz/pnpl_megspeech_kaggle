"""Compute a frozen MEG-XL trunk's features of every window a run reads, once, for `--data.features`.

A frozen trunk maps a window to the same features every epoch; a live one costs ~26 min an epoch on the full
training set, the whole set's features about one such epoch. `--read pooled` keeps the mean and max over every
channel's tokens (`--context.encoder cached`); `--read resolved` keeps each channel's own, its time-averaged token
in whitened principal components (`--context.encoder resolved`).

uv run --extra megxl python -m megspeech.training.extract --name megxl-inst --context.conv.instance
uv run --extra megxl python -m megspeech.training.extract --name megxl-channels --read resolved --context.conv.instance
"""

import argparse
import json
import logging
import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from typing import Self

import torch
from jaxtyping import Float
from torch import Tensor

from core.paths import DataRoot
from core.tree import Tree
from megspeech.data.corpus import Corpus
from megspeech.data.features import HOLDOUT, TRAIN, Features
from megspeech.data.holdout import HoldoutCache, Track
from megspeech.data.runs import CHANNELS
from megspeech.data.windows import Windows
from megspeech.model.megxl import ATTENTION_BATCH
from megspeech.model.megxl_words import COMPONENTS, Words
from megspeech.model.scaler import SCALER_SAMPLE, Scaler
from megspeech.training.config import RunConfig

LOG = logging.getLogger(__name__)

CHUNK = ATTENTION_BATCH // CHANNELS
"""Windows per forward pass: the most the trunk's temporal attention takes in one call."""
BASIS_SAMPLE = 2048
"""Windows whose channels' tokens fit the resolved read's components: 626k token vectors for 16 axes of 512."""

Reader = Callable[[Float[Tensor, "b c t"]], Float[Tensor, "b f k"]]


@dataclass
class Pooled:
    """The trunk's mean and max over every channel's tokens, one feature per window's channel axis entry."""

    words: Words

    def __call__(self, x: Float[Tensor, "b c t"]) -> Float[Tensor, "b e 1"]:
        return self.words.embed(x)[..., None]


@dataclass
class Averaged:
    """Each channel's tokens averaged over the window's time."""

    words: Words

    def __call__(self, x: Float[Tensor, "b c t"]) -> Float[Tensor, "b c d"]:
        return self.words.trunk(x).mean(2)


@dataclass
class Resolved:
    """Each channel's time-averaged token in the whitened principal components of a sample's."""

    averaged: Averaged
    mean: Float[Tensor, " d"]
    basis: Float[Tensor, "d k"]

    @classmethod
    def fit(cls, averaged: Averaged, sample: Float[Tensor, "n c d"], components: int = COMPONENTS) -> Self:
        tokens = sample.float().flatten(0, 1)
        mean = tokens.mean(0)
        _, spread, axes = torch.pca_lowrank(tokens - mean, q=components, center=False)
        return cls(averaged, mean, axes / (spread / math.sqrt(len(tokens) - 1)))

    def __call__(self, x: Float[Tensor, "b c t"]) -> Float[Tensor, "b c k"]:
        return (self.averaged(x).float() - self.mean) @ self.basis


@dataclass
class Extractor:
    """What `read` makes of windows `scaler` standardises and `words` normalises, as the sentence model reads them."""

    words: Words
    scaler: Scaler
    read: Reader
    chunk: int = CHUNK

    def __call__(self, windows: Windows) -> Float[Tensor, "n f k"]:
        rows = torch.arange(len(windows), device=windows.device)
        with torch.no_grad(), torch.autocast(windows.device.type, dtype=torch.bfloat16):
            return torch.cat(
                [self.read(self.scaled(windows(part))).to(torch.bfloat16) for part in rows.split(self.chunk)]
            )

    def scaled(self, x: Float[Tensor, "b c t"]) -> Float[Tensor, "b c t"]:
        return self.words.normalise(self.scaler(x), x.ne(0).any(dim=1, keepdim=True))


class Read(StrEnum):
    POOLED = "pooled"
    RESOLVED = "resolved"

    def reader(self, extractor: Extractor, sample: Windows) -> Reader:
        """The read, its components fitted on `sample` through `extractor` when it has any."""
        averaged = Averaged(extractor.words)
        builders: dict[Read, Callable[[], Reader]] = {
            Read.POOLED: lambda: Pooled(extractor.words),
            Read.RESOLVED: lambda: Resolved.fit(averaged, replace(extractor, read=averaged)(sample)),
        }
        return builders[self]()


def main() -> None:
    """Save the features of the training split, validation domains and Deep holdout under models/features/<name>."""
    tree = Tree(RunConfig)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--read", type=Read, choices=list(Read), default=Read.POOLED)
    tree.arguments(parser)
    args = parser.parse_args()
    config = tree.parse(args)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    root, device = DataRoot.from_config(), torch.device("cuda")
    data = replace(config.data, features="")
    corpus = Corpus.load(data, root, device)
    torch.manual_seed(config.optim.seed)
    words = Words.pretrained(root, config.context.conv).to(device).eval()
    base = Extractor(words, Scaler(corpus.train.sample(SCALER_SAMPLE)).to(device), Averaged(words))
    extract = replace(base, read=args.read.reader(base, Windows.stacked(corpus.train.sample(BASIS_SAMPLE))))
    track = Track.DEEP
    cache = HoldoutCache.of(root, data.decimate, track)
    cache.build()
    holdout = replace(cache.windows(device, data.samples, standardised=data.standardised), baseline=data.centring)
    splits = {TRAIN: corpus.train.windows} | {name: split.windows for name, split in corpus.validation.items()}
    features = Features.under(root, args.name)
    for name, windows in (splits | {f"{HOLDOUT}-{track}": holdout}).items():
        start = time.perf_counter()
        path = features.save(name, extract(windows))
        LOG.info("%s: %d windows in %.0fs -> %s", name, len(windows), time.perf_counter() - start, path)
    (features.path(TRAIN).parent / "config.json").write_text(json.dumps(asdict(config) | {"read": args.read}, indent=1))


if __name__ == "__main__":
    main()
