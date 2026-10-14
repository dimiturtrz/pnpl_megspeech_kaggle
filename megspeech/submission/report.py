"""A trained model's local test and Kaggle submissions: what `megspeech.submission` writes for a loaded model, and
what a training run asks for in its own process, reusing the geometry its cache was loaded in.

    uv run python -m megspeech.training.cli.train --name x --report.local true --report.tracks deep,broad
"""

import json
import logging
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch

from core.paths import DataRoot
from core.tree import Tree
from megspeech.data.cache.alignment import Geometry
from megspeech.data.cache.kaggle import Track
from megspeech.data.config import METRICS
from megspeech.data.raw.runs import Run
from megspeech.evaluation.exported import ExportedModel
from megspeech.evaluation.group import Group
from megspeech.evaluation.loading import Loader
from megspeech.evaluation.rulers.base import TrackRuler
from megspeech.evaluation.rulers.broad import BroadRuler
from megspeech.evaluation.rulers.deep import DeepRuler
from megspeech.evaluation.rulers.localtest import LocalTest
from megspeech.evaluation.scoring.config import Scoring
from megspeech.evaluation.scoring.probabilities import PROBS
from megspeech.model.input.subjects import Fallback, Unseen
from megspeech.submission.config import ReportConfig
from megspeech.submission.writer import KaggleWriter

LOG = logging.getLogger(__name__)

RULERS: dict[Track, type[TrackRuler]] = {Track.DEEP: DeepRuler, Track.BROAD: BroadRuler}
SUBMISSIONS = {Track.DEEP: "submission.csv", Track.BROAD: "broad_submission.csv"}
HITS = {Track.DEEP: "hits.pt", Track.BROAD: "broad_hits.pt"}
SCORING = Scoring(fallback=Fallback(Unseen.DEEP))


@dataclass(frozen=True)
class Submission:
    """`track`'s submission for a loaded model, at the adjustment the track's ruler selects; `scoring` is recorded
    beside its members so the blend can be rebuilt from the record alone."""

    track: Track
    scoring: Scoring = SCORING

    def write(self, loaded: ExportedModel, runs: list[Run], out: Path) -> Path:
        """The ruler's hits, the record and the Kaggle test rows under `out`; the submission's path."""
        ruler = RULERS[self.track](loaded, runs)
        selected = ruler.select()
        out.mkdir(parents=True, exist_ok=True)
        ruler.graded(selected.tau).save(out / HITS[self.track])
        record = {
            "members": list(loaded.names),
            "config": {"data": asdict(loaded.data), "scoring": asdict(self.scoring)},
            "best": asdict(selected),
        }
        (out / METRICS).write_text(json.dumps(record, indent=1))
        return KaggleWriter(self.track).write(loaded, selected.tau, out / SUBMISSIONS[self.track])


@dataclass(frozen=True)
class Recorded:
    """A written blend as its record left it: members, scoring, and the track whose submission sits beside them."""

    members: tuple[str, ...]
    scoring: Scoring
    track: Track

    @classmethod
    def read(cls, out: Path) -> "Recorded":
        record = json.loads((out / METRICS).read_text())
        track = next(track for track, name in SUBMISSIONS.items() if (out / name).is_file())
        scoring = Tree(Scoring, SCORING).load(record["config"]["scoring"], saved=True)
        return cls(tuple(record["members"]), scoring, track)


@dataclass(frozen=True)
class Report:
    """`config`'s reports for the run saved as `name`, each read as a separate `megspeech.submission` would, from the
    saved checkpoint, but in `geometry`, the one the training cache was loaded in."""

    config: ReportConfig
    root: DataRoot
    name: str
    device: torch.device
    geometry: Geometry
    probabilities: bool = False
    """Save the local test's per-word probabilities into the run's `probs.pt`."""

    def run(self) -> None:
        runs = Run.discover(self.root.raw())
        if self.config.local:
            saved = self.root.models() / self.name / PROBS if self.probabilities else None
            LocalTest(self.loaded(SCORING), runs).report(self.name, saved)
        for track in self.config.tracks:
            scoring = replace(SCORING, aligned=self.config.aligned and track is Track.BROAD)
            written = Submission(track, scoring).write(
                self.loaded(scoring), runs, self.root.models() / f"{self.name}-{track}"
            )
            LOG.info("wrote %s", written)

    def loaded(self, scoring: Scoring) -> ExportedModel:
        return Loader.of(Group(self.root, scoring, self.device, self.geometry), [self.name])
