"""The held-out MASC ruler (megspeech-7gf.147.2): MASC subjects a model never trained on, each read as a zero-shot
Kaggle test subject is (`Reading.UNSEEN`, through `--fallback.unseen`), scored on every labelled word of its MASC runs.

Every MASC subject heard the same four stories, so a held-out subject's text is text training heard from the others:
its unseen-span subset is empty, and the ruler scores all its labelled words instead. It measures a new BRAIN on
familiar text, not new text; that confound is shared by every model scored on it, so it compares models trained with
the same subjects held out, never against the Broad or Deep rulers. A subject's score is balanced top-10; the ruler is
the unweighted mean over subjects at one adjustment for all.
`--head word` scores the per-word MEG head, since the context head saturates at 1.0 on held-out subjects.

    uv run python -m megspeech.evaluation.rulers.heldout NAME --subjects 124 125 126 127 --fallback.unseen mean
"""

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from core.paths import DataRoot
from core.tree import Tree
from megspeech.data.config import METRICS
from megspeech.data.labels.instances import STORY
from megspeech.data.raw.runs import Run
from megspeech.device import Device
from megspeech.evaluation.loaded.ensemble import Loader
from megspeech.evaluation.loaded.group import Group
from megspeech.evaluation.loaded.model import LoadedModel
from megspeech.evaluation.scoring.config import Scoring
from megspeech.evaluation.scoring.score import Head, Score
from megspeech.model.input.subjects import Reading

LOG = logging.getLogger(__name__)
SCORING = Scoring(reading=Reading.UNSEEN)


@dataclass(frozen=True)
class Lineage:
    """The runs a model descends from: itself and, recursively, each `optim.stem` and `optim.warm` its record names."""

    models: Path

    def subjects(self, name: str, seen: frozenset[str] = frozenset()) -> set[int]:
        """The subjects `name` and every ancestor trained on."""
        config = json.loads((self.models / name / METRICS).read_text())["config"]
        optim = Tree.flat(config.get("optim", {}))
        parents = {str(optim[key]) for key in ("stem", "warm") if optim.get(key)} - seen
        own = set(config.get("data", {}).get("subjects", ()))
        return own.union(*(self.subjects(parent, seen | {name} | parents) for parent in parents))


@dataclass(frozen=True)
class HeldOutRuler:
    """The ruler over the MASC `subjects` held out of training."""

    subjects: tuple[int, ...]
    head: Head = Head.CONTEXT

    def runs(self, loaded: LoadedModel, runs: list[Run]) -> dict[int, list[Run]]:
        """Each held subject's MASC runs; an error if the model trained on any of the subjects."""
        lineage = Lineage(loaded.root.models())
        ancestry = set(loaded.data.subjects).union(*(lineage.subjects(name) for name in loaded.names))
        trained = sorted(set(self.subjects) & ancestry)
        if trained:
            raise ValueError(f"subjects {trained} are in the model's or its stem's training subjects, not held out")
        masc = [run for run in runs if STORY.fullmatch(run.key[0])]
        return {subject: [run for run in masc if run.subject == subject] for subject in self.subjects}

    def subject(self, loaded: LoadedModel, runs: list[Run]) -> Score:
        """One subject's labelled words over all of its `runs`, scored together."""
        contexts: list[Tensor] = []
        labels: list[Tensor] = []
        for _, split in loaded.splits(runs).values():
            labelled = split.labelled()
            contexts.append(loaded.evaluator.logits(loaded.model, split)[self.head][labelled])
            labels.append(split.y[labelled])
        return loaded.score(torch.cat(contexts), torch.cat(labels))

    def select(self, loaded: LoadedModel, runs: list[Run]) -> Score:
        """The macro over the held subjects with any MASC run, each subject logged at the macro's adjustment."""
        held = {subject: found for subject, found in self.runs(loaded, runs).items() if found}
        scores = {subject: self.subject(loaded, found) for subject, found in held.items()}
        macro = Score.mean(scores.values())
        for subject, score in scores.items():
            LOG.info(
                "subject %3d  runs %d  bacc10 %.4f  n %d", subject, len(held[subject]), score.by_tau[macro.tau], score.n
            )
        LOG.info(
            "held-out ruler %.4f tau %g over %d subjects, %d words  by tau %s",
            macro.bacc10,
            macro.tau,
            len(scores),
            macro.n,
            {tau: round(value, 4) for tau, value in macro.by_tau.items()},
        )
        return macro


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="+", help="trained models under models/, averaged as one")
    parser.add_argument("--subjects", nargs="+", type=int, required=True, help="MASC subjects held out of training")
    parser.add_argument("--device", default=Device.CPU, help="cpu keeps scoring off a GPU a training run owns")
    parser.add_argument("--head", type=Head, choices=list(Head), default=Head.CONTEXT, help="head to score")
    tree = Tree(Scoring, SCORING)
    tree.arguments(parser)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    root = DataRoot.from_config()
    loaded = Loader.of(Group(root, tree.parse(args), torch.device(args.device)), args.names)
    HeldOutRuler(tuple(args.subjects), args.head).select(loaded, Run.everywhere(root))


if __name__ == "__main__":
    main()
