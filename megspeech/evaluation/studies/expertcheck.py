"""Own-task check of an expert fold model (megspeech-6ow.3): on runs it held out, does its speech head's output point
at the teacher's target of the word, beyond the target of another word of the same runs?

    uv run python -m megspeech.evaluation.studies.expertcheck --expert textword --source gpt2-large --device cpu
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.nn import functional

from core.paths import DataRoot
from megspeech.data.batching.split import Split
from megspeech.data.config import DataConfig
from megspeech.data.labels.vocabulary import IGNORE, VOCAB_SIZE
from megspeech.data.raw.runs import Run
from megspeech.evaluation.reader import Reader
from megspeech.evaluation.studies.expert import Expert, Head
from megspeech.objective.targets.speech import Speech, Targets

LOG = logging.getLogger(__name__)

SEED = 0
TOP = 5
CANDIDATES = 10
"""The nearest centroids a word counts as found among: the competition metric's BAcc@10."""


@dataclass(frozen=True)
class Agreement:
    """Mean cosine of a head's words to their own targets and to a shuffled word's, and the feature spread."""

    own: float
    shuffled: float
    wins: float
    """Share of words closer to their own target than to the shuffled word's: 0.5 is chance."""
    spread: float
    words: int

    @classmethod
    def of(cls, head: Head, split: Split, part: pd.DataFrame, directory: Path) -> "Agreement":
        """`head` on `split`'s words, the words of `part` against their teacher targets under `directory`."""
        speech = Speech.gather(directory, part)
        out = head.vectors(split)
        present = speech.rows >= 0
        mine = out[present.cpu()]
        targets = functional.normalize(speech.vectors.float(), dim=-1).cpu()[speech.rows[present].cpu()]
        order = torch.randperm(len(targets), generator=torch.Generator().manual_seed(SEED))
        own, other = (mine * targets).sum(-1), (mine * targets[order]).sum(-1)
        return cls(
            float(own.mean()),
            float(other.mean()),
            float((own > other).float().mean()),
            float(out.std(0).mean()),
            len(targets),
        )


@dataclass(frozen=True)
class Centroids:
    """The mean unit target of each vocabulary word over some runs, and how lopsided those targets are."""

    means: Float[Tensor, "v d"]
    offset: float
    """Norm of the mean target vector over the typical per-dimension std: large when a common direction dominates."""
    top: float
    """Share of the targets' variance in their five leading directions."""

    @classmethod
    def fit(cls, words: pd.DataFrame, directory: Path) -> "Centroids":
        """The centroids of the vocabulary words among `words` (`run`, `position`, `target`)."""
        speech = Speech.gather(directory, words)
        present = (speech.rows >= 0).numpy()
        labels = torch.tensor(words["target"].to_numpy())
        vectors = functional.normalize(speech.vectors.float(), dim=-1)
        labels = labels[present]
        known = labels != IGNORE
        vectors, labels = vectors[known], labels[known]
        means = torch.zeros(VOCAB_SIZE, vectors.shape[1]).index_add_(0, labels, vectors)
        means = means / torch.bincount(labels, minlength=VOCAB_SIZE).clamp(min=1).unsqueeze(1)
        centred = vectors - vectors.mean(0)
        variance = torch.linalg.svdvals(centred).square()
        offset = vectors.mean(0).norm() / vectors.std(0).mean()
        return cls(means, float(offset), float(variance[:TOP].sum() / variance.sum()))

    def balanced(self, out: Float[Tensor, "n d"], labels: Int[Tensor, " n"]) -> float:
        """Balanced accuracy over the vocabulary of `out` finding its word among its `CANDIDATES` nearest centroids."""
        known = labels != IGNORE
        near = (functional.normalize(out[known], dim=-1) @ functional.normalize(self.means, dim=-1).T).topk(CANDIDATES)
        truth = labels[known]
        hits = torch.zeros(VOCAB_SIZE).index_add_(0, truth, (near.indices == truth[:, None]).any(-1).float())
        counts = torch.bincount(truth, minlength=VOCAB_SIZE)
        return float((hits[counts > 0] / counts[counts > 0]).mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expert", required=True)
    parser.add_argument("--source", required=True, help="the teacher's directory under processed/speech")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fit-runs", type=int, default=12, help="training runs the centroids average")
    parser.add_argument("--seen", action="store_true", help="runs the model trained on, not held out")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    root, device, expert = DataRoot.from_config(), torch.device(args.device), Expert(args.expert)
    name = root.models() / expert.model(args.fold)
    data = DataConfig.shared([name], DataConfig.grouping)
    held = expert.held(root, [args.fold])[0]
    runs = [run for run in Run.everywhere(root) if (run.name in held) != args.seen][: args.runs]
    head = Head.load(name, expert.tap, device)
    reader = Reader(root, data, device, geometry=data.geometry(root, device))
    directory = Targets.directory(root, args.source)
    trained = [run for run in Run.everywhere(root) if run.name not in held][: args.fit_runs]
    centroids = Centroids.fit(reader.load(trained)[1], directory)
    LOG.info("targets: mean norm / std %.2f  top-5 variance share %.3f", centroids.offset, centroids.top)
    for run, (part, split) in reader.splits(runs).items():
        LOG.info("%s %s", run.name, Agreement.of(head, split, part, directory))
        LOG.info("%s centroid bacc@10 %.4f", run.name, centroids.balanced(head.vectors(split), split.y.cpu()))


if __name__ == "__main__":
    main()
