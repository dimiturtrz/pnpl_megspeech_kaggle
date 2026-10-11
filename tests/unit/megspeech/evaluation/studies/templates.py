from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from core.paths import DataRoot
from megspeech.data.cache.run_cache import WORD_TABLE as TABLE
from megspeech.data.config import DataConfig, Val
from megspeech.data.constants import CHANNELS
from megspeech.data.labels.phonemes import Pronunciation
from megspeech.data.labels.vocabulary import PRIMARY, VOCAB_SIZE, Vocabulary
from megspeech.data.raw.runs import Partition, Run
from megspeech.evaluation.studies.templatemeans import Scale, Sums, Templates
from megspeech.evaluation.studies.templates import ABSENT, Experiment, Noise, Readout, Whitener
from megspeech.objective.targets.t5 import Prototypes
from tests.unit.megspeech.evaluation.loaded import CPU as DEVICE
from tests.unit.megspeech.evaluation.loaded import NAMES, SAMPLES, SPOKEN
from tests.unit.megspeech.evaluation.studies.templatemeans import CPU, REPS, SHAPE, WORDS, filled, windows


def test_whitener_apply() -> None:
    x = torch.randn(3, *SHAPE)
    white = Whitener(torch.eye(SHAPE[0]) * 2, torch.eye(SHAPE[1]))
    assert torch.allclose(white.apply(x), x * 2)


def test_spatial_only() -> None:
    white = Whitener(torch.eye(SHAPE[0]), torch.ones(SHAPE[1], SHAPE[1])).spatial_only()
    assert torch.equal(white.temporal, torch.eye(SHAPE[1]))


def test_noise_add() -> None:
    noise = Noise.empty(SHAPE, CPU)
    noise.add(torch.randn(8, *SHAPE))
    assert noise.seen[0] == 8 * SHAPE[1]
    assert noise.seen[1] == 8 * SHAPE[0]
    assert noise.spatial.trace() > 0


def test_noise_empty() -> None:
    noise = Noise.empty(SHAPE, CPU)
    assert noise.spatial.shape == (SHAPE[0], SHAPE[0])
    assert noise.temporal.shape == (SHAPE[1], SHAPE[1])


def test_shrunk() -> None:
    generator = torch.Generator().manual_seed(0)
    rows = torch.randn(40, 8, generator=generator, dtype=torch.float64)
    total, quartic = rows.T @ rows, rows.pow(2).sum(dim=1).pow(2).sum()
    cov = Noise.shrunk(total, quartic, torch.tensor(40.0, dtype=torch.float64))
    assert torch.allclose(cov, cov.T)
    assert cov.trace() == pytest.approx(float((total / 40).trace()), rel=1e-6)
    assert torch.linalg.eigvalsh(cov).min() > 0


def test_inverse_root() -> None:
    cov = torch.diag(torch.tensor([4.0, 9.0], dtype=torch.float64))
    root = Noise.inverse_root(cov)
    assert torch.allclose(root @ cov.float() @ root, torch.eye(2), atol=1e-5)


def test_whitener() -> None:
    noise = Noise.empty(SHAPE, CPU)
    noise.add(torch.randn(64, *SHAPE))
    white = noise.whitener()
    assert white.spatial.shape == (SHAPE[0], SHAPE[0])
    assert white.temporal.shape == (SHAPE[1], SHAPE[1])


def decoder(count: int = 30) -> tuple[Readout, Readout, torch.Tensor, torch.Tensor]:
    x, y = windows(count)
    sums = Sums.empty(1, REPS, SHAPE, CPU)
    sums.add(0, x, y)
    scale = sums.scale()
    templates = sums.templates(scale)
    noise = Noise.empty(SHAPE, CPU)
    noise.add(scale.apply(x) - templates.means[0][y])
    return Readout.correlation(templates), Readout.discriminant(templates, noise.whitener()), scale.apply(x), y


def test_correlation() -> None:
    read, _, x, y = decoder()
    assert (read.scores(x, 0).argmax(dim=1) == y).float().mean() > 0.95


def test_discriminant() -> None:
    _, read, x, y = decoder()
    assert (read.scores(x, 0).argmax(dim=1) == y).float().mean() > 0.95
    assert read.bias[0, 0] > 0


def test_readout_of() -> None:
    templates = Templates(torch.randn(1, VOCAB_SIZE, *SHAPE), torch.ones(1, VOCAB_SIZE, dtype=torch.long))
    read = Readout.of(templates, lambda x: x.flatten(1), discriminant=False)
    assert read.means.shape == (1, VOCAB_SIZE, SHAPE[0] * SHAPE[1])
    assert read.bias.sum() == 0


def test_readout_scores() -> None:
    read, _, x, _ = decoder()
    scores = read.scores(x, 0)
    assert scores.shape == (len(x), VOCAB_SIZE)
    assert (scores[:, WORDS:] == ABSENT).all()


PROTOTYPE_WIDTH = 4


TRAIN = ("sub-0_ses-1_task-Sherlock1_run-1", "sub-0_ses-2_task-Sherlock1_run-1", "sub-0_ses-3_task-TheMoth_run-1")


def experiment(tmp_path: Path) -> Experiment:
    """Subject 0's three training runs and the fixture's held-out ones, recorded and discoverable under the root."""
    root = DataRoot(tmp_path)
    cache = DataConfig(val=Val.OFFICIAL).run_cache(root).ready()
    names = (*NAMES, *TRAIN)
    onset = 0.5 + np.arange(len(SPOKEN)) * 0.5
    rows = []
    for name in names:
        np.save(cache / f"{name}.npy", np.ones((CHANNELS, SAMPLES), dtype=np.float16))
        rows.append(pd.DataFrame({"run": name, "onset": onset, "word": SPOKEN, "sentence": [0, 0, 0, 1, 1, 1]}))
        (root.raw() / "meg").mkdir(parents=True, exist_ok=True)
        (root.raw() / "events").mkdir(parents=True, exist_ok=True)
        (root.raw() / "meg" / f"{name}_meg.h5").touch()
        (root.raw() / "events" / f"{name}_events.tsv").write_text("onset\n")
    pd.concat(rows).assign(end=5.0).to_csv(cache / TABLE, sep="\t", index=False)
    return Experiment(root, DEVICE, reps=REPS)


def trained(tmp_path: Path) -> tuple[Experiment, list[Run], list[str]]:
    study = experiment(tmp_path)
    chosen = study.reader().data.trained(Run.everywhere(study.root))
    return study, chosen, sorted({DataConfig.stratum(run.key[0]) for run in chosen})


def test_reader(tmp_path: Path) -> None:
    reader = experiment(tmp_path).reader(Partition.TEST)
    assert reader.partition == Partition.TEST
    assert reader.data == DataConfig(val=Val.OFFICIAL)


def test_chunks(tmp_path: Path) -> None:
    study, chosen, _ = trained(tmp_path)
    parts = list(Experiment.chunks(study.reader(), chosen))
    assert len(parts) == len(chosen)
    assert all(len(words) == len(split.y) for _, words, split in parts)


def test_trials(tmp_path: Path) -> None:
    study, chosen, _ = trained(tmp_path)
    _, _, split = next(Experiment.chunks(study.reader(), chosen))
    batches = list(Experiment.trials(split))
    assert sum(len(y) for _, y in batches) == len(split.labelled())
    assert all(x.dtype == torch.float32 and len(x) == len(y) for x, y in batches)


def test_first(tmp_path: Path) -> None:
    study, chosen, datasets = trained(tmp_path)
    sums, durations = study.first(study.reader(), chosen, datasets)
    assert sums.counts.sum() == len(chosen) * len(SPOKEN)
    assert durations.counts.sum() == len(chosen) * (len(SPOKEN) - 2)
    assert len(sums.halves) == REPS
    with pytest.raises(ValueError, match="no training runs"):
        study.first(study.reader(), [], datasets)


def test_second(tmp_path: Path) -> None:
    study, chosen, datasets = trained(tmp_path)
    sums, _ = study.first(study.reader(), chosen, datasets)
    scale = sums.scale()
    pooled, grouped = study.second(study.reader(), chosen, datasets, scale, sums.templates(scale))
    assert pooled.seen[0] > 0
    assert grouped.seen[1] == pooled.seen[1]
    with pytest.raises(ValueError, match="no training runs"):
        study.second(study.reader(), [], datasets, scale, sums.templates(scale))


def readable(count: int = 40) -> tuple[Templates, tuple[Noise, Noise], Scale]:
    x, y = windows(count)
    sums = Sums.empty(1, REPS, SHAPE, CPU)
    sums.add(0, x, y)
    scale = sums.scale()
    templates = sums.templates(scale)
    noise = Noise.empty(SHAPE, CPU)
    noise.add(scale.apply(x) - templates.means[0][y])
    return templates, (noise, noise), scale


def test_readouts() -> None:
    templates, noises, _ = readable()
    assert set(Experiment.readouts(templates, noises)) == {
        "corr/pooled",
        "corr/dataset",
        "lda-spatial/pooled",
        "lda-kron/pooled",
        "lda-spatial/dataset",
        "lda-kron/dataset",
    }
    assert Experiment.readouts(templates, noises)["lda-kron/dataset"][1]
    assert not Experiment.readouts(templates, noises)["corr/pooled"][1]


def test_decode(tmp_path: Path) -> None:
    study, chosen, datasets = trained(tmp_path)
    reader = study.reader()
    sums, _ = study.first(reader, chosen, datasets)
    scale = sums.scale()
    templates = sums.templates(scale)
    readouts = Experiment.readouts(templates, study.second(reader, chosen, datasets, scale, templates))
    lines = study.decode(Partition.VALIDATION, datasets, scale, readouts)
    assert len(lines) == len(readouts)
    assert all(line.startswith("validation decode ") and "pooled=" in line for line in lines)


def test_experiment_scores(tmp_path: Path) -> None:
    study, chosen, datasets = trained(tmp_path)
    sums, _ = study.first(study.reader(), chosen, datasets)
    scale = sums.scale()
    _, _, split = next(Experiment.chunks(study.reader(), chosen))
    scores = study.scores(split, scale, Readout.correlation(sums.templates(scale)), 0)
    assert scores.shape == (len(split.labelled()), VOCAB_SIZE)


def test_balanced() -> None:
    scores = torch.eye(VOCAB_SIZE)[:WORDS]
    labels = torch.arange(WORDS)
    assert Experiment.balanced(scores, labels) == pytest.approx(1.0)


def test_reliability() -> None:
    sums = filled(40)
    lines = experiment_lines(sums)
    assert lines[0].startswith("reliability pooled words=5")
    assert len(lines) == 3


def experiment_lines(sums: Sums) -> list[str]:
    return Experiment(DataRoot(Path()), DEVICE, reps=REPS).reliability(sums, sums.scale(), ["a", "b"])


def test_content(tmp_path: Path) -> None:
    study, chosen, datasets = trained(tmp_path)
    sums, durations = study.first(study.reader(), chosen, datasets)
    spelled = torch.from_numpy(Vocabulary.token(pd.Series(PRIMARY.words))).sort().values
    Pronunciation(spelled, torch.zeros(VOCAB_SIZE, 4, dtype=torch.long)).save(Pronunciation.path(study.root))
    vectors = torch.from_numpy(np.random.default_rng(0).normal(size=(VOCAB_SIZE, PROTOTYPE_WIDTH)).astype(np.float32))
    Prototypes(vectors).save(Prototypes.path(study.root))
    lines = study.content(sums.templates(sums.scale()), durations, datasets)
    assert len(lines) == 3 * (1 + len(datasets))
    assert lines[0].startswith("rsa pooled vs duration r=")


def test_run(tmp_path: Path) -> None:
    study = experiment(tmp_path)
    spelled = torch.from_numpy(Vocabulary.token(pd.Series(PRIMARY.words))).sort().values
    Pronunciation(spelled, torch.zeros(VOCAB_SIZE, 4, dtype=torch.long)).save(Pronunciation.path(study.root))
    vectors = torch.from_numpy(np.random.default_rng(0).normal(size=(VOCAB_SIZE, PROTOTYPE_WIDTH)).astype(np.float32))
    Prototypes(vectors).save(Prototypes.path(study.root))
    lines = list(study.run())
    assert lines[0].startswith("train runs=")
    assert any(line.startswith("test decode") for line in lines)
