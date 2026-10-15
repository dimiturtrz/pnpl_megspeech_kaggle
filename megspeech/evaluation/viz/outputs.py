"""One model's outputs on the sample set: per-sample predictions and embedding, and its own metric on the set."""

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
import torch
from jaxtyping import Float, Int
from torch import Tensor

from core.paths import DataRoot
from megspeech.data.cache.kaggle import Track
from megspeech.data.config import METRICS
from megspeech.data.corpus import Corpus
from megspeech.data.labels.vocabulary import PRIMARY
from megspeech.data.raw.runs import DEEP, Column, Run
from megspeech.device import Device
from megspeech.evaluation.loaded.ensemble import Loader
from megspeech.evaluation.loaded.exported import ExportedModel
from megspeech.evaluation.loaded.group import Group
from megspeech.evaluation.metric.ruler import TOP, Graded, Ruler
from megspeech.evaluation.scoring.config import Scoring
from megspeech.evaluation.scoring.probabilities import Words
from megspeech.evaluation.viz.embed import PROJECTORS, Embedder
from megspeech.evaluation.viz.records import KEY

METRIC = f"macro bAcc@{TOP}, unseen-span words"
"""The selection macro (`Ruler.point`): each counted domain's balanced top-10 accuracy over its unseen-span words."""


@dataclass(frozen=True)
class Outputs:
    """A model's `models/<name>.json` and its `manifest.json` entry."""

    name: str
    track: str
    score: float | None
    projections: dict[str, dict[str, list[float]]]
    samples: dict[str, dict[str, object]]

    def manifest(self) -> dict[str, object]:
        return {
            "name": self.name,
            "track": self.track,
            "labels": list(PRIMARY.words),
            "metric": METRIC,
            "score": self.score,
        }

    def write(self, directory: Path) -> Path:
        """`<directory>/<name>.json`."""
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.name}.json"
        document = {"name": self.name, "projections": self.projections, "samples": self.samples}
        path.write_text(json.dumps(document), encoding="utf-8")
        return path


@dataclass(frozen=True)
class Predictions:
    """Per-sample records from probabilities, labels and each projector's embedding points, rows aligned with `ids`."""

    ruler: Ruler = Ruler()

    def records(
        self,
        ids: Sequence[str],
        probabilities: Float[Tensor, "n v"],
        labels: Int[Tensor, " n"],
        points: dict[str, Float[np.ndarray, "n 2"]],
    ) -> dict[str, dict[str, object]]:
        hits = self.ruler.hits(probabilities, labels)
        truth = probabilities.gather(1, labels[:, None])[:, 0]
        top = probabilities.topk(self.ruler.k, dim=1)
        return {
            name: {
                "pred": PRIMARY.words[int(top.indices[i, 0])],
                "correct": bool(hits[i]),
                "p_true": float(truth[i]),
                "error": 1 - float(truth[i]),
                "topk": [[PRIMARY.words[int(c)], float(p)] for c, p in zip(top.indices[i], top.values[i], strict=True)],
                "emb": {projector: [float(x) for x in at[i]] for projector, at in points.items()},
            }
            for i, name in enumerate(ids)
        }


@dataclass(frozen=True)
class Located:
    """Each sample's domain and word position there, from the `KEY` columns of the samples and the domains' words."""

    domain: Sequence[str]
    position: Int[np.ndarray, " n"]

    @classmethod
    def of(cls, keys: pd.DataFrame, rows: dict[str, pd.DataFrame]) -> "Located":
        """`keys`' rows found in `rows` (each domain's words in its split's order)."""
        tables = [
            table[KEY].reset_index(drop=True).rename_axis("position").reset_index().assign(domain=name)
            for name, table in rows.items()
        ]
        found = keys[KEY].merge(pd.concat(tables), how="left", on=KEY, validate="one_to_one")
        if found["position"].isna().any():
            raise KeyError(keys[found["position"].isna().to_numpy()].head().to_string())
        return cls(found["domain"].tolist(), found["position"].to_numpy(dtype=np.int64))

    def at(self, name: str) -> Int[np.ndarray, " k"]:
        """The samples in domain `name`, in sample order."""
        return np.flatnonzero(np.asarray(self.domain) == name)

    def sampled(self, graded: Graded, novel: dict[str, Int[Tensor, " m"]]) -> Graded:
        """`graded` kept to the sampled words: each domain's hits on its `novel` words that are samples."""
        kept = {}
        for name, (hits, labels) in graded.domains.items():
            mask = torch.isin(novel[name].cpu(), torch.from_numpy(self.position[self.at(name)]))
            if mask.any():
                kept[name] = (hits.cpu()[mask], labels.cpu()[mask])
        return Graded(kept)


class Reading(Protocol):
    def read(self, root: DataRoot, name: str, runs: list[Run], keys: pd.DataFrame) -> Outputs: ...


@dataclass(frozen=True)
class Reader:
    """Loads each model on `device` and reads it on the sample set."""

    device: str = Device.CPU

    @staticmethod
    def tau(root: DataRoot, name: str) -> float:
        """The adjustment val selected for `name` (`metrics.json` `best.macro.tau`)."""
        saved = json.loads((root.models() / name / METRICS).read_text(encoding="utf-8"))
        return float(saved["best"]["macro"]["tau"])

    @staticmethod
    def track(loaded: ExportedModel) -> str:
        return str(Track.DEEP if tuple(loaded.data.subjects) == (DEEP,) else Track.BROAD)

    def read(  # devtools-ignore: test-mirror
        self, root: DataRoot, name: str, runs: list[Run], keys: pd.DataFrame
    ) -> Outputs:
        """`name` on every domain of `runs`, kept to the samples in `keys` (`KEY` columns and `id`)."""
        loaded = Loader.of(Group(root, Scoring(), torch.device(self.device)), [name])
        source, words = loaded.load(runs)
        held: dict[str, list[Run]] = {}
        for run in runs:
            held.setdefault(loaded.data.domain(run), []).append(run)
        splits = {
            domain: loaded.read(domain, split)
            for domain, split in Corpus.domains(source, words, held, loaded.data).items()
        }
        rows = {domain: words[words[Column.RUN].isin([r.name for r in group])] for domain, group in held.items()}
        located = Located.of(keys, rows)
        evaluator, calibration = loaded.evaluator, loaded.calibration(self.tau(root, name))
        logits = {domain: evaluator.logits(loaded.model, split) for domain, split in splits.items()}
        embedder = Embedder(evaluator)
        size = len(keys)
        probabilities = torch.zeros(size, len(PRIMARY.words))
        labels = torch.zeros(size, dtype=torch.int64)
        embedded: dict[str, Tensor] = {}
        for domain, split in splits.items():
            mine = located.at(domain)
            if not len(mine):
                continue
            chosen = Words.of(evaluator, logits[domain], split, calibration)
            row = torch.searchsorted(chosen.ids, torch.from_numpy(located.position[mine]))
            probabilities[mine] = chosen.probabilities[row]
            labels[mine] = chosen.labels[row]
            embedded[domain] = embedder.embedded(loaded.model, split)[located.position[mine]].cpu()
        dims = max((e.shape[1] for e in embedded.values()), default=0)
        emb = torch.zeros(size, dims)
        for domain, values in embedded.items():
            emb[located.at(domain)] = values
        projections = {projector.name: projector.project(emb.numpy()) for projector in PROJECTORS}
        counted = evaluator.eligible(splits)
        graded = evaluator.grade({d: evaluator.head(logits[d]) for d in counted}, counted, calibration)
        novel = {d: split.novel(evaluator.seen) for d, split in counted.items()}
        kept = located.sampled(graded, novel)
        score = Ruler().point(kept) if kept.domains else math.nan
        return Outputs(
            name,
            self.track(loaded),
            None if math.isnan(score) else score,
            {name: {"explained": p.explained.tolist()} for name, p in projections.items()},
            Predictions().records(
                keys["id"].tolist(), probabilities, labels, {name: p.points for name, p in projections.items()}
            ),
        )
