"""Every row's head log-probabilities from an ensemble, out-of-vocabulary rows included, saved beside it so LM decoding
reads each word's span neighbours' posteriors without loading the model again.

`probs.pt` keeps only the labelled words; a Kaggle test sentence epoch holds every word heard (its onsets are 0.21 s
apart at the median, as all words are in the training text, 0.53 s for the 50 words alone), so decoding needs the
posteriors of the rows no label counts. Sections: `val` and `local-test` per scored domain, `kaggle` the Deep track's
rows in canonical order.

    uv run python -m megspeech.evaluation.studies.lmrows m11xsp7hub6d2v3
"""

import argparse
import json
import logging
from dataclasses import dataclass, replace

import torch
from jaxtyping import Float, Int
from torch import Tensor

from core.paths import DataRoot
from megspeech.data.batching.split import Split
from megspeech.data.cache.kaggle import Track
from megspeech.data.config import METRICS
from megspeech.data.raw.runs import Partition, Run
from megspeech.device import Device
from megspeech.evaluation.loaded.ensemble import Loader
from megspeech.evaluation.loaded.exported import ExportedModel
from megspeech.evaluation.loaded.group import Group
from megspeech.evaluation.scoring.probabilities import SECTIONS
from megspeech.evaluation.scoring.score import Head
from megspeech.evaluation.studies.crowding import VAL
from megspeech.evaluation.studies.neighbourfusion import DEEP

LOG = logging.getLogger(__name__)

ROWS = "lmrows.pt"
KAGGLE = "kaggle"
"""The section of the Deep Kaggle test's rows, held as one domain of this name."""


@dataclass(frozen=True)
class Posteriors:
    """One domain's every row: both heads' log-probabilities, its label (`IGNORE` outside the vocabulary or unknown),
    and its spans' row indices (`PAD` past a span's end)."""

    head: Float[Tensor, "n v"]
    word: Float[Tensor, "n v"]
    labels: Int[Tensor, " n"]
    index: Int[Tensor, "g w"]

    @classmethod
    def of(cls, split: Split, logits: dict[Head, Float[Tensor, "n v"]]) -> "Posteriors":
        return cls(
            logits[Head.CONTEXT].float().cpu(),
            logits[Head.WORD].float().cpu(),
            split.y.cpu(),
            split.index.cpu(),
        )

    def record(self) -> dict[str, Tensor]:
        return {"head": self.head, "word": self.word, "labels": self.labels, "index": self.index}

    @classmethod
    def restored(cls, record: dict[str, Tensor]) -> "Posteriors":
        return cls(record["head"], record["word"], record["labels"], record["index"])

    @staticmethod
    def load(path: str, section: str) -> dict[str, "Posteriors"]:  # devtools-ignore: test-mirror
        saved = torch.load(path)[section]
        return {name: Posteriors.restored(record) for name, record in saved.items()}


@dataclass(frozen=True)
class Dump:
    """The loaded ensemble's rows of every section."""

    loaded: ExportedModel

    def section(self, partition: Partition) -> dict[str, Posteriors]:  # devtools-ignore: test-mirror
        reader = replace(self.loaded, partition=partition)
        splits, logits = reader.domain_logits(Run.discover(reader.root.raw()))
        return {name: Posteriors.of(split, logits[name]) for name, split in splits.items()}

    def kaggle(self) -> dict[str, Posteriors]:  # devtools-ignore: test-mirror
        loaded = self.loaded
        evaluator = loaded.evaluator
        split = loaded.kaggle(Track.DEEP)
        return {KAGGLE: Posteriors.of(split, evaluator.logits(loaded.model, split))}

    @staticmethod
    def main() -> None:  # devtools-ignore: test-mirror
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("name", help="an ensemble under models/ whose metrics.json lists its members")
        parser.add_argument("--device", default=Device.CUDA)
        args = parser.parse_args()
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
        root = DataRoot.from_config()
        out = root.models() / args.name
        members = json.loads((out / METRICS).read_text())["members"]
        dump = Dump(Loader.of(Group(root, DEEP, torch.device(args.device)), members))
        saved = {
            VAL: dump.section(Partition.VALIDATION),
            SECTIONS[Partition.TEST]: dump.section(Partition.TEST),
            KAGGLE: dump.kaggle(),
        }
        records = {
            section: {name: rows.record() for name, rows in domains.items()} for section, domains in saved.items()
        }
        torch.save(records, out / ROWS)
        LOG.info("wrote %s", out / ROWS)


if __name__ == "__main__":
    Dump.main()
