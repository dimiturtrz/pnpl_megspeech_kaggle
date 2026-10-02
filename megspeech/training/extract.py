"""Compute a frozen MEG-XL trunk's pooled features of every window a run reads, once, for `--data.features`.

A frozen trunk maps a window to the same features every epoch; a live one costs ~26 min an epoch on the full
training set, the whole set's features about one such epoch.

uv run --extra megxl python -m megspeech.training.extract --name megxl-inst --context.conv.instance
"""

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass, replace

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
from megspeech.model.megxl_words import Words
from megspeech.model.scaler import SCALER_SAMPLE, Scaler
from megspeech.training.config import RunConfig

LOG = logging.getLogger(__name__)

CHUNK = ATTENTION_BATCH // CHANNELS
"""Windows per forward pass: the most the trunk's temporal attention takes in one call."""


@dataclass
class Extractor:
    """The pooled features `words` reads from windows `scaler` standardises, as the sentence model would read them."""

    words: Words
    scaler: Scaler
    chunk: int = CHUNK

    def __call__(self, windows: Windows) -> Float[Tensor, "n e"]:
        rows = torch.arange(len(windows), device=windows.device)
        with torch.no_grad(), torch.autocast(windows.device.type, dtype=torch.bfloat16):
            return torch.cat([self.pooled(windows(part)) for part in rows.split(self.chunk)])

    def pooled(self, x: Float[Tensor, "b c t"]) -> Float[Tensor, "b e"]:
        scaled = self.words.normalise(self.scaler(x), x.ne(0).any(dim=1, keepdim=True))
        return self.words.embed(scaled).to(torch.bfloat16)


def main() -> None:
    """Save the features of the training split, validation domains and Deep holdout under models/features/<name>."""
    tree = Tree(RunConfig)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    tree.arguments(parser)
    args = parser.parse_args()
    config = tree.parse(args)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    root, device = DataRoot.from_config(), torch.device("cuda")
    data = replace(config.data, features="")
    corpus = Corpus.load(data, root, device)
    torch.manual_seed(config.optim.seed)
    words = Words.pretrained(root, config.context.conv).to(device).eval()
    extract = Extractor(words, Scaler(corpus.train.sample(SCALER_SAMPLE)).to(device))
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
    (features.path(TRAIN).parent / "config.json").write_text(json.dumps(asdict(config), indent=1))


if __name__ == "__main__":
    main()
