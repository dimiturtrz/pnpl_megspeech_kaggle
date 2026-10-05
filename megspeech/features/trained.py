"""Save a trained run's own CNN word embeddings as features, its encoder frozen, for `--context.encoder trained`.

The first stage of a two-stage model: a context model trained on these reads exactly what the run's encoder made of
each word, so the encoder and the context model can be trained and compared apart (megspeech-7gf.50).

uv run python -m megspeech.features.trained --run sent40-ioi-s0 --name ioi-s0
"""

import argparse
import logging
from dataclasses import asdict, dataclass

import torch
from jaxtyping import Float
from torch import Tensor

from core.paths import DataRoot
from megspeech.data.batching.split import Split
from megspeech.data.config import LocalRuler
from megspeech.features.extract import CHUNK, Export
from megspeech.model.sentence import SentenceNet
from megspeech.training.config import RunConfig


@dataclass
class Trained:
    """`net`'s word embedding of each of a split's windows, `chunk` windows a forward pass, in bfloat16 and shaped
    as one feature per channel like MEG-XL's pooled read."""

    net: SentenceNet
    chunk: int = CHUNK

    def __call__(self, split: Split) -> Float[Tensor, "n e 1"]:
        rows = torch.arange(len(split.windows), device=split.subject.device)
        with torch.no_grad(), torch.autocast(split.subject.device.type, dtype=torch.bfloat16):
            parts = [self.net.embed(split.windows(part), split.subject[part]) for part in rows.split(self.chunk)]
        return torch.cat(parts).to(torch.bfloat16)[..., None]


def main() -> None:
    """Save `--run`'s embeddings of the training split, validation domains and Deep Kaggle test under
    models/features/<name>, the run's own data settings choosing the words."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--ruler", type=LocalRuler, choices=list(LocalRuler), default=LocalRuler.OFFICIAL)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    root, device = DataRoot.from_config(), torch.device("cuda")
    run = root.models() / args.run
    config = RunConfig.saved(run)
    data = config.data.under(args.ruler)
    net = SentenceNet.load(run / "model.pt").to(device).eval()
    Export.of(data, root, args.name, device).save(Trained(net), asdict(config) | {"run": args.run})


if __name__ == "__main__":
    main()
