"""How well a model's own tag head (`--context.conv.head tagged`) names each official validation word's part of
speech, scored on the words whose in-context tag is known, with `syntax`'s probes' scores (all, in-50, outside, per
domain) and per-tag recall.

    uv run python -m megspeech.evaluation.studies.tagacc posfac-s0
"""

import argparse
import logging
from dataclasses import dataclass

import numpy as np
import torch
from jaxtyping import Int

from core.paths import DataRoot
from megspeech.data.config import DataConfig
from megspeech.data.labels.context_tags import ContextTags
from megspeech.data.labels.phonemes import UNKNOWN
from megspeech.data.raw.runs import Run
from megspeech.evaluation.reader import Reader
from megspeech.evaluation.studies.syntax import Embedded, Probed

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class Headed:
    """Every scored word's in-context tag, the head's prediction, membership in the 50 and domain."""

    tags: Int[np.ndarray, " n"]
    predicted: Int[np.ndarray, " n"]
    vocabulary: np.ndarray
    domain: np.ndarray

    def scored(self) -> tuple[Probed, Int[np.ndarray, " k"]]:
        """The known-tag words as `syntax`'s `Probed` and the head's prediction on them: a word the tagger could not
        align has no tag to score."""
        known = self.tags != UNKNOWN
        return Probed(self.tags[known], self.vocabulary[known], self.domain[known]), self.predicted[known]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", help="a trained tagged-head model under models/")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    root, device = DataRoot.from_config(), torch.device("cuda")
    trained_in = root.models() / args.name
    data = DataConfig.shared([trained_in], DataConfig.grouping)
    reader = Reader(root, data, device, geometry=data.geometry(root, device, aligned=False))
    contexts = ContextTags.load(ContextTags.path(root))
    embedded = Embedded.load(trained_in / "model.pt", device)
    tags: list[np.ndarray] = []
    predicted: list[np.ndarray] = []
    vocabulary: list[np.ndarray] = []
    domain: list[np.ndarray] = []
    for name, runs in reader.ruled(Run.discover(root.raw())).items():
        for words, split in reader.splits(runs).values():
            tags.append(contexts.classes(words["run"].to_numpy(), words["onset"].to_numpy(np.float64)))
            predicted.append(embedded.tagged(split))
            vocabulary.append((words["target"] >= 0).to_numpy())
            domain.append(np.full(len(words), name))
    headed = Headed(*map(np.concatenate, (tags, predicted, vocabulary, domain)))
    scored, hit = headed.scored()
    LOG.info(
        "scoring %d known-tag words of %d, majority-tag accuracy %.4f",
        len(hit),
        len(headed.tags),
        scored.majority(),
    )
    scores = scored.scores(hit)
    LOG.info("head (bacc, acc, n) %s", {k: (round(b, 4), round(a, 4), n) for k, (b, a, n) in scores.items()})
    LOG.info("head recall %s", {k: (round(r, 3), n) for k, (r, n) in scored.recalls(hit).items()})


if __name__ == "__main__":
    main()
