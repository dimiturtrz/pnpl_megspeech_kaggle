"""The Broad ruler with Sinkhorn balancing over all subjects' unseen-span words at once, against the ruler's own
per-subject balancing, from the same logits.

The Kaggle Broad submission balances its test rows pooled across subjects; the ruler balances each subject alone. Both
numbers read hits per subject and take the macro as `BroadRuler` does, so the gap is the balancing's scope alone.

    uv run python -m megspeech.evaluation.studies.pooledbalance sent40-s0 sent40-s1 --aligned [--fallback.unseen mean]
"""

import argparse
import logging
from dataclasses import dataclass

import torch
from jaxtyping import Float, Int
from torch import Tensor

from core.paths import DataRoot
from core.tree import Tree
from megspeech.data.raw.runs import Run
from megspeech.device import Device
from megspeech.evaluation.group import Group
from megspeech.evaluation.loaded import LoadedModel
from megspeech.evaluation.loading import Loader
from megspeech.evaluation.scoring.config import Scoring
from megspeech.evaluation.scoring.evaluator import Evaluator
from megspeech.evaluation.scoring.score import ADJUSTMENTS, Score
from megspeech.model.input.subjects import Fallback
from megspeech.model.input.subjects import Unseen as SensorMap

LOG = logging.getLogger(__name__)

SCORING = Scoring(fallback=Fallback(SensorMap.DEEP))
"""The submission's scoring defaults (`megspeech.submission.report.SCORING`), which this layer may not import."""

GROUPS = {"1-12": range(1, 13), "13-22": range(13, 23), "23-32": range(23, 33)}
"""The subject groups the ruler is read by."""

ByTau = dict[float, dict[int, float]]


@dataclass(frozen=True)
class Unseen:
    """One subject's unseen-span words: the context head's log-probabilities and the labels."""

    logits: Float[Tensor, "n v"]
    y: Int[Tensor, " n"]


@dataclass(frozen=True)
class PooledBalance:
    evaluator: Evaluator
    log_prior: Float[Tensor, " v"]

    def alone(self, parts: dict[int, Unseen]) -> ByTau:
        """Per adjustment, each subject's balanced top-10 with the balancing fit on that subject's words alone."""
        return {
            tau: {
                s: self.bacc(self.evaluator.calibration(self.log_prior, tau).adjusted(p.logits), p.y)
                for s, p in parts.items()
            }
            for tau in ADJUSTMENTS
        }

    def pooled(self, parts: dict[int, Unseen]) -> ByTau:
        """Per adjustment, each subject's balanced top-10 with the balancing fit on every subject's words together."""
        subjects = list(parts)
        sizes = [len(parts[s].y) for s in subjects]
        logits = torch.cat([parts[s].logits for s in subjects])
        out: ByTau = {}
        for tau in ADJUSTMENTS:
            adjusted = self.evaluator.calibration(self.log_prior, tau).adjusted(logits).split(sizes)
            out[tau] = {s: self.bacc(a, parts[s].y) for s, a in zip(subjects, adjusted, strict=True)}
        return out

    def bacc(self, adjusted: Float[Tensor, "n v"], y: Int[Tensor, " n"]) -> float:
        ruler = self.evaluator.ruler
        return ruler.balanced(ruler.hits(adjusted, y), y)

    @staticmethod
    def macro(by_tau: ByTau) -> Score:
        """The ruler: subjects' unweighted mean at each adjustment, the best adjustment chosen once for all."""
        return Score.best({tau: Score.average(v.values()) for tau, v in by_tau.items()}, 0)

    @staticmethod
    def grouped(by_tau: ByTau, tau: float) -> dict[str, float]:
        return {
            name: Score.average(v for s, v in by_tau[tau].items() if s in subjects) for name, subjects in GROUPS.items()
        }

    def report(self, parts: dict[int, Unseen]) -> str:
        lines = []
        for label, by_tau in (("per-subject", self.alone(parts)), ("pooled", self.pooled(parts))):
            macro = self.macro(by_tau)
            means = {tau: round(Score.average(v.values()), 4) for tau, v in by_tau.items()}
            groups = {name: round(v, 4) for name, v in self.grouped(by_tau, macro.tau).items()}
            lines.append(f"{label}: ruler {macro.bacc10:.4f} tau {macro.tau:g} by tau {means} groups {groups}")
        return "\n".join(lines)


class Parts:
    @staticmethod
    def read(loaded: LoadedModel, runs: list[Run]) -> dict[int, Unseen]:
        """Each counted subject's unseen-span words, read once as `BroadRuler.select` reads them."""
        parts = {}
        for run in loaded.broad(runs):
            split, context = loaded.run_context(run)
            novel = split.novel(loaded.evaluator.seen)
            if len(novel) >= loaded.evaluator.min_novel:
                parts[run.subject] = Unseen(context[novel], split.y[novel])
        return parts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="+", help="trained models under models/, averaged as one")
    parser.add_argument("--device", default=Device.CUDA)
    tree = Tree(Scoring, SCORING)
    tree.arguments(parser)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    root = DataRoot.from_config()
    loaded = Loader.of(Group(root, tree.parse(args), torch.device(args.device)), args.names)
    parts = Parts.read(loaded, Run.discover(root.raw()))
    LOG.info("%d subjects\n%s", len(parts), PooledBalance(loaded.evaluator, loaded.log_prior).report(parts))


if __name__ == "__main__":
    main()
