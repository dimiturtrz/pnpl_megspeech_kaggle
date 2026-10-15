"""What each source of a fused model contributes: the model scored with every source on, then with each source in turn
silenced (its own encoder, "primary", and every branch), tau fixed at the all-on model's (megspeech-7gf.131).

Scored as `ioi` scores: the same domains, tau and macro on both partitions.

    uv run python -m megspeech.evaluation.studies readout.ablation sent40-fused-s0 --grouping kaggle --partition test
"""

import argparse
import logging
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass, replace

from torch import nn

from core.paths import DataRoot
from core.tree import Tree
from megspeech.data.batching.split import Split
from megspeech.data.raw.runs import Partition, Run
from megspeech.device import Device
from megspeech.evaluation.loaded.ensemble import Loader
from megspeech.evaluation.loaded.group import Group
from megspeech.evaluation.loaded.model import LoadedModel
from megspeech.evaluation.scoring.config import Scoring
from megspeech.model.encoder.fusion import Fusion
from megspeech.model.input.subjects import Fallback, Unseen

LOG = logging.getLogger(__name__)

ALL_ON = "none"


@dataclass(frozen=True)
class Sources:
    """The fusions of a loaded model's members, silenced together."""

    fusions: tuple[Fusion, ...]

    @classmethod
    def of(cls, model: nn.Module) -> "Sources":
        """The fusions inside `model`; a model with no branches has nothing to ablate."""
        fusions = tuple(m for m in model.modules() if isinstance(m, Fusion))
        if not any(f.branches for f in fusions):
            raise ValueError("the model has no fusion branches: train with --context.fused")
        if len({f.names() for f in fusions}) != 1:
            raise ValueError(f"members fuse different sources: {[f.names() for f in fusions]}")
        return cls(fusions)

    def names(self) -> tuple[str, ...]:
        """The sources, the model's own first."""
        return self.fusions[0].names()

    def silenced(self, name: str) -> ExitStack:
        """Within the block `name` weighs 0 in every member."""
        stack = ExitStack()
        for fusion in self.fusions:
            stack.enter_context(fusion.silenced(name))
        return stack

    def sweep(self, measure: Callable[[], str]) -> dict[str, str]:
        """`measure` with every source on, then with each silenced in turn, by arm."""
        arms = {ALL_ON: measure()}
        for name in self.names():
            with self.silenced(name):
                arms[name] = measure()
        return arms


@dataclass(frozen=True)
class Ablation:
    """A loaded fused model with the domains of both partitions read once."""

    loaded: LoadedModel
    domains: dict[Partition, dict[str, Split]]
    tau: float

    @classmethod
    def of(cls, loaded: LoadedModel, runs: list[Run]) -> "Ablation":  # devtools-ignore: test-mirror
        domains = {}
        for partition in (Partition.VALIDATION, Partition.TEST):
            held = replace(loaded, partition=partition)
            domains[partition], _ = held.domains(held.ruled(runs))
        return cls(loaded, domains, loaded.report(runs).tau)

    def macros(self) -> str:  # devtools-ignore: test-mirror
        """Each partition's macro of the model as it now stands."""
        loaded = self.loaded
        return " ".join(
            f"{partition} macro {loaded.evaluator.point(loaded.model, split, loaded.calibration(self.tau)):.4f}"
            for partition, split in self.domains.items()
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="+", help="trained fused models under models/, averaged as one")
    tree = Tree(Scoring, Scoring(fallback=Fallback(Unseen.DEEP)))
    tree.arguments(parser)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    root = DataRoot.from_config()
    loaded = Loader.of(Group(root, tree.parse(args), Device.CUDA.as_torch()), args.names)
    sources = Sources.of(loaded.model)
    ablation = Ablation.of(loaded, Run.discover(root.raw()))
    LOG.info("tau %s", ablation.tau)
    for arm, line in sources.sweep(ablation.macros).items():
        LOG.info("silenced=%s %s", arm, line)


if __name__ == "__main__":
    main()
