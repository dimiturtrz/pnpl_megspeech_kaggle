import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from core.paths import DataRoot
from megspeech.data.raw.runs import Run
from megspeech.evaluation.viz.export import MANIFEST, MODELS, VIZ, ExportConfig, Exporter
from megspeech.evaluation.viz.outputs import Outputs
from megspeech.evaluation.viz.records import SAMPLES, SampleSet
from tests.unit.megspeech.data.raw.runs import events, meg

HELD = "sub-0_ses-11_task-Sherlock1_run-2"
TRAIN = "sub-0_ses-1_task-Sherlock1_run-1"
POOLED = "sub-12_ses-11_task-Sherlock1_run-1"


def test_runs(tmp_path: Path) -> None:
    root = DataRoot(tmp_path)
    for name in (HELD, TRAIN, POOLED):
        events(Run(meg(root.raw(), name)))
    meg(root.raw(), "sub-0_ses-12_task-Sherlock1_run-1")
    assert [run.name for run in Exporter.runs(root)] == [HELD]


@dataclass(frozen=True)
class Drawn:
    def draw(self, root: DataRoot, config: ExportConfig, runs: list[Run]) -> SampleSet:
        keys = pd.DataFrame({"run": [HELD], "onset": [1.0], "id": ["0000"]})
        return SampleSet(keys, [{"id": "0000", "runs": len(runs)}], {"0000": np.zeros((2, 3), dtype=np.float32)})


@dataclass(frozen=True)
class Read:
    def read(self, root: DataRoot, name: str, runs: list[Run], keys: pd.DataFrame) -> Outputs:
        return Outputs(
            name,
            "deep",
            0.5,
            {"pca": {"explained": [0.3, 0.2]}},
            {i: {"p_true": 0.5} for i in keys["id"]},
            {},
            torch.ones(len(keys), 1),
        )


@dataclass(frozen=True)
class Layout:
    def describe(self, root: DataRoot) -> dict[str, object]:
        return {"names": ["MEG0111"]}


def test_run(tmp_path: Path) -> None:
    root = DataRoot(tmp_path)
    events(Run(meg(root.raw(), HELD)))
    out = Exporter(root, Drawn(), Read(), Layout()).run(ExportConfig(("m1", "m2"), n=1, seed=3, name="set"))
    assert out == root.kind(VIZ) / "set"
    manifest = json.loads((out / MANIFEST).read_text(encoding="utf-8"))
    assert [m["name"] for m in manifest["models"]] == ["m1", "m2"]
    assert (manifest["name"], manifest["seed"], manifest["samples"]) == ("set", 3, SAMPLES)
    assert manifest["sensors"] == {"names": ["MEG0111"]}
    assert json.loads((out / SAMPLES).read_text(encoding="utf-8")) == [{"id": "0000", "runs": 1}]
    assert json.loads((out / MODELS / "m2.json").read_text(encoding="utf-8"))["samples"] == {"0000": {"p_true": 0.5}}
    assert (out / "meg" / "0000.f32").stat().st_size == 2 * 3 * 4
