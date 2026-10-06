"""What the per-epoch validation histories of past runs say about stopping early and about ranking recipes sooner.

Every complete, val-selected run under the models root is grouped by its recipe (epochs, EMA on or off, deep-only or
multi-subject, which val). Per group: where the best epoch falls and what patience stops would have saved and lost
(a), how well val at epoch k ranks the runs against their final selected val (b), and the spread of the selected val
across runs differing only in `optim.seed` (c). Selected val only: a run's local test score is not in its record
(megspeech-iuh).

    uv run python -m megspeech.evaluation.studies.screening
"""

import argparse
import json
import logging
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from jaxtyping import Float, Int

from core.paths import DataRoot
from megspeech.data.config import DEEP, METRICS
from megspeech.training.config import RunConfig

LOG = logging.getLogger(__name__)

PATIENCES = (3, 5, 8)
HORIZONS = (5, 10, 15, 20, 25)
QUANTILES = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
TOP = (1, 3)
MIN_GROUP = 6
MIN_SEEDS = 2
PRECISION = 4


@dataclass(frozen=True)
class Run:
    """One finished run: its recipe family, the config minus seed, its seed and its val at every epoch."""

    name: str
    family: str
    recipe: str
    seed: int
    val: Float[np.ndarray, " e"]

    @classmethod
    def of(cls, directory: Path) -> "Run | None":
        """The run in `directory`, or nothing when it has no complete val-selected history under a loadable config:
        every epoch, or fewer with the record's `stopped` mark (early stop), unlike a crash."""
        record = directory / METRICS
        if not record.exists():
            return None
        saved = json.loads(record.read_text())
        history = saved.get("history")
        try:
            config = RunConfig.saved(directory)
        except (KeyError, ValueError):
            return None
        if not history or config.keeps_last or not all("macro" in row for row in history):
            return None
        if len(history) != config.optim.epochs and not (
            saved.get("stopped") is True and len(history) < config.optim.epochs
        ):
            return None
        seedless = replace(config, optim=replace(config.optim, seed=0))
        return cls(
            directory.name,
            cls.family_of(config),
            repr(seedless),
            config.optim.seed,
            cls.padded([row["macro"]["bacc10"] for row in history], config.optim.epochs),
        )

    @staticmethod
    def padded(val: list[float], epochs: int) -> Float[np.ndarray, " e"]:
        """`val` carried to `epochs` by its best so far: what an early-stopped run keeps over the epochs it skipped."""
        curve = np.array(val)
        return np.concatenate([curve, np.full(epochs - len(curve), curve.max())])

    @staticmethod
    def family_of(config: RunConfig) -> str:
        """Epochs, smoothing, subjects and val kind: the recipe axes whose runs share a schedule."""
        subjects = "deep" if tuple(config.data.subjects) == (DEEP,) else "multi"
        smoothing = "ema" if config.optim.ema > 0 else "raw"
        return f"e{config.optim.epochs:02d}-{smoothing}-{subjects}-{config.data.val.value}"


@dataclass(frozen=True)
class Family:
    """The runs of one recipe family, their val histories stacked: runs by epochs."""

    name: str
    runs: tuple[Run, ...]

    @property
    def curves(self) -> Float[np.ndarray, "n e"]:
        return np.stack([run.val for run in self.runs])

    @property
    def best(self) -> Float[np.ndarray, " n"]:
        """The selected val of every run."""
        return self.curves.max(axis=1)

    def best_epochs(self) -> Int[np.ndarray, " n"]:
        """The 1-based epoch each run selects."""
        return self.curves.argmax(axis=1) + 1

    def epoch_quantiles(self) -> pd.Series:
        """Quantiles of the selected epoch."""
        return pd.Series(np.quantile(self.best_epochs(), QUANTILES), index=[f"q{q:g}" for q in QUANTILES])

    @staticmethod
    def stops(curves: Float[np.ndarray, "n e"], patience: int) -> Int[np.ndarray, " n"]:
        """The 0-based epoch each run stops at: `patience` epochs past its last strict improvement, else its last."""
        running = np.maximum.accumulate(curves, axis=1)
        improved = np.concatenate([np.ones((len(curves), 1), bool), curves[:, 1:] > running[:, :-1]], axis=1)
        epochs = np.arange(curves.shape[1])
        since = epochs - np.maximum.accumulate(np.where(improved, epochs, 0), axis=1)
        stalled = since >= patience
        return np.where(stalled.any(axis=1), stalled.argmax(axis=1), curves.shape[1] - 1)

    def patience(self, patience: int) -> pd.Series:
        """Mean epochs saved and mean/max val lost to the selected best, had every run stopped at `patience`."""
        curves = self.curves
        stop = self.stops(curves, patience)
        kept = np.maximum.accumulate(curves, axis=1)[np.arange(len(curves)), stop]
        lost = self.best - kept
        return pd.Series(
            {
                "saved": float((curves.shape[1] - 1 - stop).mean()),
                "lost_mean": float(lost.mean()),
                "lost_max": float(lost.max()),
                "stopped": float((stop < curves.shape[1] - 1).mean()),
            }
        )

    @staticmethod
    def leaders(early: Float[np.ndarray, " n"], final: Float[np.ndarray, " n"], top: int) -> bool:
        """Whether the final leader is among the early `top`."""
        return bool(final.argmax() in np.argsort(-early)[:top])

    def agreement(self, horizon: int, *, best_so_far: bool) -> pd.Series:
        """Rank agreement of val at epoch `horizon` (or its best so far) with the selected val."""
        curves = self.curves
        early = np.maximum.accumulate(curves, axis=1)[:, horizon - 1] if best_so_far else curves[:, horizon - 1]
        pair = pd.DataFrame({"early": early, "final": self.best})
        row = {
            "spearman": pair["early"].corr(pair["final"], method="spearman"),
            "kendall": pair["early"].corr(pair["final"], method="kendall"),
        }
        row.update({f"top{top}": float(self.leaders(early, self.best, top)) for top in TOP})
        return pd.Series(row)

    def horizons(self, *, best_so_far: bool) -> pd.DataFrame:
        """`agreement` at every horizon the schedule reaches."""
        epochs = self.curves.shape[1]
        rows = {k: self.agreement(k, best_so_far=best_so_far) for k in HORIZONS if k < epochs}
        return pd.DataFrame(rows).T.rename_axis("k")


class Screening:
    """The families of a models root, and the tables over them."""

    def __init__(self, runs: Sequence[Run], min_group: int = MIN_GROUP) -> None:
        self.runs = tuple(runs)
        grouped = self.grouped(self.runs, lambda run: run.family)
        self.families = tuple(
            Family(name, tuple(grouped[name])) for name in sorted(grouped) if len(grouped[name]) >= min_group
        )

    @staticmethod
    def grouped(runs: Sequence[Run], key: Callable[[Run], str]) -> dict[str, list[Run]]:
        """`runs` by `key`, in first-seen order."""
        groups: dict[str, list[Run]] = {}
        for run in runs:
            groups.setdefault(key(run), []).append(run)
        return groups

    @classmethod
    def discover(cls, models: Path, min_group: int = MIN_GROUP) -> "Screening":
        """The screening over every run directory under `models`."""
        found = (Run.of(directory) for directory in sorted(models.iterdir()) if directory.is_dir())
        return cls([run for run in found if run is not None], min_group)

    def epochs(self) -> pd.DataFrame:
        """(a) the selected-epoch quantiles per family."""
        rows = {f.name: pd.concat([pd.Series({"n": len(f.runs)}), f.epoch_quantiles()]) for f in self.families}
        return pd.DataFrame(rows).T

    def patiences(self) -> pd.DataFrame:
        """(a) what each patience stop saves and costs, per family."""
        rows = {(f.name, p): f.patience(p) for f in self.families for p in PATIENCES}
        return pd.DataFrame(rows).T.rename_axis(["family", "patience"])

    def rankings(self, *, best_so_far: bool) -> pd.DataFrame:
        """(b) rank agreement per family and horizon."""
        frames = {f.name: f.horizons(best_so_far=best_so_far) for f in self.families}
        return pd.concat(frames, names=["family"])

    def seed_groups(self) -> pd.DataFrame:
        """(c) the selected val of every set of runs differing only in seed."""
        rows = []
        for family in self.families:
            for group in self.grouped(family.runs, lambda run: run.recipe).values():
                if len({run.seed for run in group}) >= MIN_SEEDS:
                    values = np.array([run.val.max() for run in group])
                    rows.append((family.name, group[0].name, len(group), values.mean(), values.std(ddof=1)))
        return pd.DataFrame(rows, columns=["family", "first_run", "n", "mean", "sd"])

    @staticmethod
    def pooled(groups: pd.DataFrame) -> pd.Series:
        """The within-group SD pooled over seed groups: sqrt(sum((n-1) sd^2) / sum(n-1))."""
        dof = groups["n"] - 1
        pooled = np.sqrt((dof * groups["sd"] ** 2).sum() / dof.sum())
        return pd.Series({"groups": len(groups), "runs": int(groups["n"].sum()), "pooled_sd": float(pooled)})

    def seed_scatter(self) -> pd.DataFrame:
        """(c) the pooled seed SD per family and over all."""
        groups = self.seed_groups()
        by_family = {name: self.pooled(part) for name, part in groups.groupby("family")}
        by_family["ALL"] = self.pooled(groups)
        return pd.DataFrame(by_family).T

    def report(self) -> str:
        """Every table, as text."""
        sections = {
            "(a) selected epoch (1-based) quantiles": self.epochs(),
            "(a) patience stops vs selected best": self.patiences(),
            "(b) rank agreement, raw val at epoch k vs selected val": self.rankings(best_so_far=False),
            "(b) rank agreement, best-so-far val by epoch k vs selected val": self.rankings(best_so_far=True),
            "(c) seed groups": self.seed_groups(),
            "(c) pooled seed SD": self.seed_scatter(),
        }
        counts = f"{len(self.runs)} complete runs, {len(self.families)} families of >= min-group"
        body = (f"## {title}\n{table.round(PRECISION).to_string()}" for title, table in sections.items())
        return "\n\n".join([counts, *body]) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-group", type=int, default=MIN_GROUP)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    models = DataRoot.from_config().models()
    LOG.info("scanning %s", models)
    sys.stdout.write(Screening.discover(models, args.min_group).report())


if __name__ == "__main__":
    main()
