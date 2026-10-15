import json
from pathlib import Path

import numpy as np
import pytest

from megspeech.evaluation.viz.agreement import Agreement, Saliencies

REGIONS = {"Left-temporal": [0], "Right-temporal": [1], "Vertex": [2]}
KINDS = ["grad", "grad", "mag"]


def saliencies(maps: np.ndarray, correct: np.ndarray | None = None) -> Saliencies:
    m, n, c = maps.shape
    rng = np.random.default_rng(1)
    return Saliencies(
        [f"m{i}" for i in range(m)],
        maps,
        (np.arange(n) % 2 == 0)[None].repeat(m, axis=0) if correct is None else correct,
        np.arange(n),
        rng.random((n, c)),
        REGIONS,
        KINDS,
    )


def replicated(m: int = 3, n: int = 40) -> Saliencies:
    base = np.random.default_rng(0).random((n, 3)) + 0.1
    return saliencies(np.stack([base / base.sum(axis=1, keepdims=True)] * m))


def test_unit() -> None:
    rows = Agreement.unit(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert rows[0] == pytest.approx([0.6, 0.8])
    assert rows[1] == pytest.approx([0.0, 0.0])


def test_pairs() -> None:
    z = np.stack([np.eye(2)] * 2)
    keep = np.ones(2, dtype=bool)
    assert Agreement.pairs(z, np.arange(2), keep) == pytest.approx(1.0)
    assert Agreement.pairs(z, np.array([1, 0]), keep) == pytest.approx(0.0)


def test_residuals() -> None:
    z = Agreement(replicated()).residuals()
    assert np.linalg.norm(z, axis=2) == pytest.approx(1.0)


def test_ratios() -> None:
    maps = np.tile(np.array([0.5, 0.25, 0.25]), (1, 2, 1))
    ratios = Agreement(saliencies(maps)).ratios()
    assert ratios["Left-temporal"] == pytest.approx(1.5)
    assert ratios["grad"] == pytest.approx(1.125)
    assert ratios["mag"] == pytest.approx(0.75)


def test_replication() -> None:
    report = Agreement(replicated()).replication()
    assert report["same_word"] == pytest.approx(1.0)
    assert report["other_sentence"] < report["same_word"]
    assert -1 <= report["amplitude"] <= 1


def test_per_model() -> None:
    assert Agreement(replicated()).per_model() == pytest.approx({"m0": 1.0, "m1": 1.0, "m2": 1.0})


def test_errors() -> None:
    maps = np.array([[[0.6, 0.2, 0.2], [0.1, 0.1, 0.8]]])
    errors = Agreement(saliencies(maps, np.array([[True, False]]))).errors()
    assert errors == pytest.approx({"correct": 0.8, "wrong": 0.2})


def test_report() -> None:
    assert set(Agreement(replicated()).report()) == {"ratios", "replication", "temporal", "models"}


def test_span(tmp_path: Path) -> None:
    np.arange(6, dtype=np.float32).reshape(2, 3).tofile(tmp_path / "x.f32")
    sample: dict[str, object] = {
        "meg": {"file": "x.f32", "channels": 2, "times": 3, "t0": 0.0, "sfreq": 1.0},
        "target_t": [1.0, 3.0],
    }
    assert Saliencies.span(tmp_path, sample) == pytest.approx([1.5, 4.5])


def test_read(tmp_path: Path) -> None:
    (tmp_path / "models").mkdir()
    (tmp_path / "meg").mkdir()
    np.ones((3, 2), dtype=np.float32).tofile(tmp_path / "meg" / "0.f32")
    meg = {"file": "meg/0.f32", "channels": 3, "times": 2, "t0": 0.0, "sfreq": 1.0}
    samples = [{"id": "0", "label": "we", "sentence": "s", "meg": meg, "target_t": [0.0, 1.0]}]
    manifest = {
        "models": [{"name": "a", "saliency": {"file": "a.saliency.u8"}}],
        "sensors": {"regions": REGIONS, "kinds": KINDS},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "samples.json").write_text(json.dumps(samples))
    (tmp_path / "models" / "a.json").write_text(json.dumps({"samples": {"0": {"pred": "we"}}}))
    np.array([255, 0, 255], dtype=np.uint8).tofile(tmp_path / "models" / "a.saliency.u8")
    read = Saliencies.read(tmp_path)
    assert read.maps[0, 0] == pytest.approx([0.5, 0.0, 0.5])
    assert read.correct.tolist() == [[True]]
    assert read.amplitude.shape == (1, 3)
