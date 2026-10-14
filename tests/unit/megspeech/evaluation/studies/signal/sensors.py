import logging
from pathlib import Path

import mne
import numpy as np
import pytest

from megspeech.data.cache.run_cache import RunCache
from megspeech.data.raw.runs import Run
from megspeech.data.raw.sources.masc import Neuromag
from megspeech.evaluation.studies.signal.sensors import (
    BANDS,
    DISTANCES,
    Coherence,
    Coherencies,
    Health,
    Selection,
    Sensors,
    Study,
)

RATE = 50.0
SAMPLES = 3000
LAG = 3
POSITIONS = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.5, 0.0]])
NAMES = ("MEG0111", "MEG0112", "MEG0113", "MEG0121")
TYPES = ("mag", "mag", "grad", "grad")
BAND = 1
"""4-8 Hz, where a 3-sample lag at 50 Hz turns the phase by about 2.3 rad."""


def recording() -> np.ndarray:
    """Sensor 0 a source, 1 its negative, 2 a lagged copy of it, 3 flat."""
    source = np.random.default_rng(0).standard_normal(SAMPLES)
    return np.stack([source, -source, np.roll(source, LAG), np.zeros(SAMPLES)]).astype(np.float32)


def sensors() -> Sensors:
    return Sensors(NAMES, TYPES, POSITIONS)


def coherence() -> Coherence:
    return Coherence(RATE, POSITIONS)


def run(name: str) -> Run:
    return Run(Path(f"sub-{name}_meg.h5"))


def test_summary() -> None:
    assert Health(("a",), ("b", "c"), ()).summary() == "dead=1 noisy=2 copies=0"


def test_listing() -> None:
    assert Health(("a",), (), ("c",)).listing() == "dead=['a'] noisy=[] copies=['c']"


def test_sensors_of() -> None:
    one = Sensors.of(Neuromag(mne.create_info(list(NAMES), RATE, list(TYPES))))
    assert one.names == NAMES
    assert one.types == TYPES


def test_ratios() -> None:
    """Each channel against its own type's median: the two mags are equal, the flat grad is far below its twin."""
    ratio = sensors().ratios(np.array([2.0, 2.0, 4.0, 0.0]))
    assert ratio[:2].tolist() == [1.0, 1.0]
    assert ratio[2] == pytest.approx(2.0)
    assert ratio[3] == 0.0


def test_pick() -> None:
    assert sensors().pick(np.array([True, False, False, True])) == (NAMES[0], NAMES[3])


def test_health() -> None:
    """A flat channel is dead, a x1000 channel noisy, a negated one a copy."""
    signal = recording()
    wide = np.concatenate([signal, 1000 * signal[:1], signal[:1]])
    many = Sensors((*NAMES, "MEG0131", "MEG0132"), ("mag", "mag", "grad", "grad", "mag", "mag"), np.zeros((6, 3)))
    health = many.health(wide, wide)
    assert health.dead == (NAMES[3],)
    assert "MEG0131" in health.noisy
    assert NAMES[0] in health.copies
    assert NAMES[2] not in health.copies


def test_spectra() -> None:
    assert coherence().spectra(recording()).shape == (4, SAMPLES // 100, 51)


def test_masks() -> None:
    masks = coherence().masks()
    assert masks.shape == (len(BANDS), 51)
    assert not (masks[:-1] & masks[1:]).any()


def test_coherency() -> None:
    """Sensors with gains +1 and -1 have |real| 1 and imag 0; a lagged copy has an imaginary part."""
    one = coherence()
    coherency = one.coherency(one.spectra(recording()), np.array([[0, 1], [0, 2]]))
    assert np.abs(coherency[0].real) == pytest.approx(1.0, abs=1e-5)
    assert coherency[0].real.max() < 0
    assert np.abs(coherency[0].imag).max() < 1e-5
    assert np.abs(coherency[1, BAND].imag) > 0.5


def test_distances() -> None:
    assert coherence().distances(np.array([[0, 1], [0, 2], [0, 3]])).tolist() == pytest.approx([0.0, 0.1, 0.5])


def test_draw() -> None:
    pairs = Coherence.draw(10, 500, 0)
    assert pairs.shape == (500, 2)
    assert (pairs[:, 0] != pairs[:, 1]).all()
    assert np.array_equal(pairs, Coherence.draw(10, 500, 0))


def test_coherencies_medians() -> None:
    values = np.array([[0.5 + 0.1j, -0.5 + 0.2j], [-0.3 - 0.3j, 0.7 + 0.0j], [0.4 + 0.2j, 0.1 + 0.1j]])
    values = np.tile(values, (1, 2))
    medians = Coherencies(values).medians()
    assert medians.shape == (4, 2)
    assert medians[0].tolist() == pytest.approx([0.4, 0.2])


def test_study_medians(caplog: pytest.LogCaptureFixture) -> None:
    values = np.tile(np.array([[0.5 + 0.1j]]), (3, 4))
    with caplog.at_level(logging.INFO):
        Study.medians("label", values)
    assert caplog.text.count("label band") == len(BANDS)


def test_by_distance() -> None:
    values = np.array([[1.0 + 0j], [0.5 + 0j], [0.2 + 0j]])
    table = Coherencies(values).by_distance(np.array([0.0, 0.01, 0.11]))
    assert table.shape == (1, len(DISTANCES) - 1)
    assert table[0, 0] == pytest.approx(0.75)
    assert table[0, 3] == pytest.approx(0.2)
    assert np.isnan(table[0, 1])


def test_selection_of() -> None:
    """Only subject 0's training runs are eligible: not its validation or test runs, nor another subject's."""
    runs = [
        run("0_ses-1_task-Sherlock1_run-1"),
        run("0_ses-2_task-Sherlock1_run-1"),
        run("0_ses-3_task-Sherlock1_run-1"),
        run("0_ses-11_task-Sherlock1_run-2"),
        run("1_ses-1_task-Sherlock1_run-1"),
    ]
    chosen = Selection.of(runs, 2).runs
    assert [r.name for r in chosen] == ["sub-0_ses-1_task-Sherlock1_run-1", "sub-0_ses-3_task-Sherlock1_run-1"]


def test_central() -> None:
    signal = np.arange(2 * 1000, dtype=np.float32).reshape(2, 1000)
    middle = Selection.central(signal, RATE, 0.2)
    assert middle.shape == (2, 600)
    assert middle[0, 0] == 200
    assert Selection.central(signal, RATE, 60.0).shape == (2, 1000)


def test_session(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    cache = RunCache(tmp_path)
    one = run("0_ses-1_task-Sherlock1_run-1")
    np.save(cache.path(one.name), recording())
    pairs = np.array([[0, 1], [0, 2]])
    with caplog.at_level(logging.INFO):
        coherency = Study(cache, sensors(), pairs, 5.0).session(one)
    assert coherency.shape == (2, len(BANDS))
    assert "dead=1" in caplog.text


def test_pooled(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    cache = RunCache(tmp_path)
    runs = [run(f"0_ses-{i}_task-Sherlock1_run-1") for i in (1, 2)]
    for one in runs:
        np.save(cache.path(one.name), recording())
    with caplog.at_level(logging.INFO):
        Study(cache, sensors(), np.array([[0, 1], [0, 2], [0, 3]]), 5.0).pooled(runs)
    assert "pooled(2)" in caplog.text
    assert "pairs per bin" in caplog.text
