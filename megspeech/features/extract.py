"""Compute a frozen MEG-XL trunk's features of every word a run reads, once, for `--data.features`.

A frozen trunk maps a window to the same features every epoch; a live one costs ~26 min an epoch on the full
training set, the whole set's features about one such epoch. `--read pooled` keeps the mean and max over every
channel's tokens of the word's own window (`--context.encoder cached`); `--read resolved` keeps each channel's own,
its time-averaged token in whitened principal components (`--context.encoder resolved`); `--read spanned` reads
the same per-channel components from the word's tokens within its whole span's continuous segment, the long
context MEG-XL was pretrained on (`--context.encoder resolved`).

uv run --extra megxl python -m megspeech.features.extract --name megxl-inst --context.conv.instance
uv run --extra megxl python -m megspeech.features.extract --name megxl-spans --read spanned --context.conv.instance
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
from jaxtyping import Float, Int
from torch import Tensor

from core.paths import DataRoot
from core.tree import Tree
from megspeech.data.batching.split import KAGGLE_FEATURES, TRAIN, Features, Split
from megspeech.data.batching.windows import Windows
from megspeech.data.cache.kaggle import Track
from megspeech.data.config import DataConfig
from megspeech.data.constants import CHANNELS, RATE
from megspeech.data.corpus import Corpus
from megspeech.data.spans.spans import PAD
from megspeech.device import Device
from megspeech.model.input.scaler import SCALER_SAMPLE, Scaler
from megspeech.model.words.megxl import COMPONENTS, MegxlWords
from megspeech.model.words.spanned import SpanWords
from megspeech.model.words.trunk import ATTENTION_BATCH, DOWNSAMPLING
from megspeech.training.config import RunConfig

LOG = logging.getLogger(__name__)

CHUNK = ATTENTION_BATCH // CHANNELS
"""Windows per forward pass: the most the trunk's temporal attention takes in one call."""
BASIS_SAMPLE = 2048
"""Windows whose channels' tokens fit the resolved read's components: 626k token vectors for 16 axes of 512."""
BASIS_SPANS = 256
"""Spans whose words fit the spanned read's components: ~1.7k words, 535k channel vectors for 16 axes of 512."""
SEGMENT = 37 * DOWNSAMPLING
"""Samples of a span's segment: the longest Kaggle test sentence epoch can run 8 onset gaps of 0.92 s plus a 1.36 s
tail, 8.7 s, rounded up to whole tokens."""
WORD = RATE // 2
"""Samples after its onset whose tokens make a word's feature: MEG-XL's own word read."""
SEGMENT_CHUNK = CHUNK * RATE // SEGMENT
"""Segments per forward pass: as many samples as a chunk of 1-s windows, the most that fit in VRAM; a full chunk of
segments spills to host memory through the driver's fallback."""

Reader = Callable[[Float[Tensor, "b c t"]], Float[Tensor, "b f k"]]
Extraction = Callable[[Split], Float[Tensor, "n f k"]]


@dataclass
class Pooled:
    """The trunk's mean and max over every channel's tokens, one feature per window's channel axis entry."""

    words: MegxlWords

    def __call__(self, x: Float[Tensor, "b c t"]) -> Float[Tensor, "b e 1"]:
        return self.words.embed(x)[..., None]


@dataclass
class Averaged:
    """Each channel's tokens averaged over the window's time."""

    words: MegxlWords

    def __call__(self, x: Float[Tensor, "b c t"]) -> Float[Tensor, "b c d"]:
        return self.words.trunk(x).mean(2)


@dataclass
class Projection:
    """Token vectors in the whitened principal components of a sample's."""

    mean: Float[Tensor, " d"]
    basis: Float[Tensor, "d k"]

    @classmethod
    def fit(cls, sample: Float[Tensor, "n c d"], components: int = COMPONENTS) -> Self:
        tokens = sample.float().flatten(0, 1)
        mean = tokens.mean(0)
        _, spread, axes = torch.pca_lowrank(tokens - mean, q=components, center=False)
        return cls(mean, axes / (spread / math.sqrt(len(tokens) - 1)))

    def __call__(self, x: Float[Tensor, "*b d"]) -> Float[Tensor, "*b k"]:
        return (x.float() - self.mean) @ self.basis


@dataclass
class Resolved:
    """Each channel's time-averaged token, projected."""

    averaged: Averaged
    projection: Projection

    def __call__(self, x: Float[Tensor, "b c t"]) -> Float[Tensor, "b c k"]:
        return self.projection(self.averaged(x))


@dataclass
class Extractor:
    """What `read` makes of windows `scaler` standardises and `words` normalises, as the sentence model reads them."""

    words: MegxlWords
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

    def tokens(self, x: Float[Tensor, "b c t"]) -> Float[Tensor, "b c k d"]:
        """The trunk's every token of `x`, scaled."""
        return self.words.trunk(self.scaled(x))


@dataclass
class Windowed:
    """A window read applied to a split's own windows."""

    extractor: Extractor

    def __call__(self, split: Split) -> Float[Tensor, "n f k"]:
        return self.extractor(split.windows)


@dataclass
class Segments:
    """Each word's tokens over its first `WORD` samples, out of the trunk's one pass over its span's continuous
    segment from the span's first onset, averaged per channel and then `project`ed."""

    extractor: Extractor
    project: Callable[[Float[Tensor, "*b d"]], Float[Tensor, "*b k"]]
    length: int = SEGMENT
    chunk: int = SEGMENT_CHUNK

    def __call__(self, split: Split) -> Float[Tensor, "n c k"]:
        spans = torch.arange(len(split.index), device=split.index.device)
        words, features = self.pooled(split, spans)
        out = torch.zeros((len(split.windows), *features.shape[1:]), dtype=features.dtype, device=features.device)
        out[words] = features
        return out

    def pooled(self, split: Split, spans: Int[Tensor, " g"]) -> tuple[Int[Tensor, " m"], Float[Tensor, "m c k"]]:
        """The words of `spans` and their features, in span order."""
        segments = replace(split.windows, length=self.length, offset=0, baseline=0)
        parts = [self.part(segments, split.index[part]) for part in spans.split(self.chunk)]
        return torch.cat([words for words, _ in parts]), torch.cat([features for _, features in parts])

    def part(self, segments: Windows, index: Int[Tensor, "b w"]) -> tuple[Int[Tensor, " m"], Float[Tensor, "m c k"]]:
        with torch.no_grad(), torch.autocast(segments.device.type, dtype=torch.bfloat16):
            tokens = self.extractor.tokens(segments(index[:, 0]))
            pooled = torch.einsum("bckd,bwk->bwcd", tokens, self.weights(segments.at, index, tokens.shape[2]))
        valid = index != PAD
        return index[valid], self.project(pooled[valid]).to(torch.bfloat16)

    @staticmethod
    def weights(at: Int[Tensor, " n"], index: Int[Tensor, "b w"], tokens: int) -> Float[Tensor, "b w k"]:
        """Equal weights on the tokens overlapping each word's first `WORD` samples into its span's segment."""
        return SpanWords.weights(at[index] - at[index[:, :1]], tokens, WORD)


class Read(StrEnum):
    POOLED = "pooled"
    RESOLVED = "resolved"
    SPANNED = "spanned"

    def extraction(self, extractor: Extractor, sample: Split) -> Extraction:
        """The read, its components fitted on `sample` through `extractor` when it has any."""
        averaged = Averaged(extractor.words)
        builders: dict[Read, Callable[[], Extraction]] = {
            Read.POOLED: lambda: Windowed(replace(extractor, read=Pooled(extractor.words))),
            Read.RESOLVED: lambda: Windowed(
                replace(
                    extractor,
                    read=Resolved(
                        averaged,
                        Projection.fit(replace(extractor, read=averaged)(Windows.stacked(sample.sample(BASIS_SAMPLE)))),
                    ),
                )
            ),
            Read.SPANNED: lambda: Segments(extractor, Projection.fit(self.spanned(extractor, sample))),
        }
        return builders[self]()

    @staticmethod
    def spanned(extractor: Extractor, sample: Split) -> Float[Tensor, "m c d"]:
        """Unprojected spanned features of `BASIS_SPANS` of `sample`'s spans, drawn without replacement."""
        spans = torch.randperm(len(sample.index), device=sample.index.device)[:BASIS_SPANS]
        return Segments(extractor, lambda x: x).pooled(sample, spans)[1]


@dataclass
class Export:
    """The MEG of the training split, every validation domain and the Deep Kaggle test, whose extracted features are
    saved under models/features/<name>, one file per split, beside the settings that made them."""

    corpus: Corpus
    kaggle: Split
    features: Features

    @classmethod
    def of(cls, data: DataConfig, root: DataRoot, name: str, device: torch.device) -> Self:
        """The splits `data` reads, their MEG rather than any features it names, in `data`'s own geometry."""
        meg = replace(data, features="")
        track = Track.DEEP
        corpus = Corpus.load(meg, root, device)
        windows = Corpus.kaggle(root, meg, track, device, corpus.geometry)
        return cls(corpus, Split.from_kaggle(track.kaggle(root.raw()), windows), Features.under(root, name))

    def splits(self) -> dict[str, Split]:
        """Every split by the name its features are saved and read under."""
        kaggle = {f"{KAGGLE_FEATURES}-{Track.DEEP}": self.kaggle}
        return {TRAIN: self.corpus.train} | self.corpus.validation | kaggle

    def save(self, extract: Extraction, settings: dict[str, object]) -> None:
        for name, split in self.splits().items():
            start = time.perf_counter()
            path = self.features.save(name, extract(split))
            LOG.info("%s: %d words in %.0fs -> %s", name, len(split.windows), time.perf_counter() - start, path)
        (self.features.path(TRAIN).parent / "config.json").write_text(json.dumps(settings, indent=1))


def main() -> None:
    """Save the features of the training split, validation domains and Deep Kaggle test under models/features/<name>."""
    tree = Tree(RunConfig)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--read", type=Read, choices=list(Read), default=Read.POOLED)
    tree.arguments(parser)
    args = parser.parse_args()
    config = tree.parse(args)
    if config.data.projected:
        parser.error("MEG-XL reads the 306 sensors; a projected config keeps fewer")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    root, device = DataRoot.from_config(), Device.CUDA.as_torch()
    export = Export.of(config.data, root, args.name, device)
    train = export.corpus.train
    torch.manual_seed(config.optim.seed)
    words = MegxlWords.pretrained(root, config.context.conv).to(device).eval()
    base = Extractor(words, Scaler(train.sample(SCALER_SAMPLE)).to(device), Pooled(words))
    export.save(args.read.extraction(base, train), asdict(config) | {"read": args.read})


if __name__ == "__main__":
    main()
