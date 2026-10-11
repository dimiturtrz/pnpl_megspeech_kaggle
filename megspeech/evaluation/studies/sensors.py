"""Sensor health and cross-sensor coherency of subject 0's training Deep recordings, the physics check of the
channel-first encoder (megspeech-7gf.125).

MEG is quasi-static: every sensor reads a real-gain mixture of the same few sources at the same instant,
`sensor_i(t) = sum_k L_ik s_k(t)`, with no propagation lag. The prediction is therefore that the complex coherency
between two sensors has an imaginary part near zero (the noise floor of the estimate) in every band, a real part that
carries the sign of the mixing, and a magnitude that falls as the sensors move apart. A lag between sensors, which
the channel-first encoder's shared temporal stack would have to absorb, shows as a large imaginary part.

Two readouts, per cached 50 Hz session:

1. Health: each channel's variance against the median variance of its sensor type (magnetometer or gradiometer);
   a ratio under `DEAD` is flat, over `NOISY` is noisy; a channel whose correlation with another reaches `COPY` is a
   linear copy of it.
2. Coherency of `PAIRS` random sensor pairs in `BANDS`: median |real| and |imag| per band, and the median
|coherency| per sensor-distance bin, pooled over the sessions.

The cap: only subject 0's `Partition.TRAIN` runs are read (no validation, test, pooled or other-subject run, nor the
Kaggle test), `SESSIONS` of them spread evenly through the sorted list so the tasks mix, and health reads each in full
(one pass of 306 variances). Coherency and copies read the central `MINUTES` of each: 2 s Hann segments give
`MINUTES * 30` averages per band bin, so the estimate's noise floor sits far under the effects looked for. CPU only,
numpy.

    uv run python -m megspeech.evaluation.studies.sensors
"""

import argparse
import logging
from dataclasses import dataclass

import numpy as np
from jaxtyping import Bool, Complex, Float, Int

from core.paths import DataRoot
from megspeech.data.cache.run_cache import RunCache
from megspeech.data.raw.fetch import NEUROMAG
from megspeech.data.raw.masc import Neuromag
from megspeech.data.raw.runs import DEEP, Partition, Run

LOG = logging.getLogger(__name__)

SESSIONS = 4
MINUTES = 5.0
PAIRS = 3000
SEGMENT = 100
"""Samples of one coherency segment: 2 s at the cache's 50 Hz, a 0.5 Hz resolution."""
BANDS = ((1.0, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 25.0))
DEAD = 1e-2
NOISY = 1e2
COPY = 0.9999
DISTANCES = (0.0, 0.03, 0.06, 0.1, 0.15, 0.2, 1.0)
"""Edges in metres of the sensor-distance bins; the first holds a magnetometer and its gradiometers at one site."""
CHUNK = 250
SECONDS = 60.0


@dataclass(frozen=True)
class Health:
    """The channels of one session by what is wrong with them."""

    dead: tuple[str, ...]
    noisy: tuple[str, ...]
    copies: tuple[str, ...]

    def summary(self) -> str:
        return f"dead={len(self.dead)} noisy={len(self.noisy)} copies={len(self.copies)}"

    def listing(self) -> str:
        return f"dead={list(self.dead)} noisy={list(self.noisy)} copies={list(self.copies)}"


@dataclass(frozen=True)
class Sensors:
    """The 306 channels in the cache's order: their names, types and device-frame positions."""

    names: tuple[str, ...]
    types: tuple[str, ...]
    positions: Float[np.ndarray, "c 3"]

    @classmethod
    def of(cls, helmet: Neuromag) -> "Sensors":
        return cls(tuple(helmet.names), tuple(helmet.types), helmet.positions)

    def ratios(self, variance: Float[np.ndarray, " c"]) -> Float[np.ndarray, " c"]:
        """Each channel's variance over the median variance of the channels of its type."""
        kinds = np.array(self.types)
        medians = {kind: np.median(variance[kinds == kind]) for kind in set(self.types)}
        return variance / np.array([medians[kind] for kind in self.types])

    def pick(self, flagged: Bool[np.ndarray, " c"]) -> tuple[str, ...]:
        return tuple(name for name, flag in zip(self.names, flagged, strict=True) if flag)

    def health(self, signal: Float[np.ndarray, "c t"], reference: Float[np.ndarray, "c r"]) -> Health:
        """Dead and noisy channels from `signal`'s variances, and the linear copies among `reference`'s channels."""
        ratio = self.ratios(signal.astype(np.float32).var(axis=1))
        with np.errstate(invalid="ignore"):
            correlation = np.nan_to_num(np.corrcoef(reference.astype(np.float32)))
        return Health(
            self.pick(ratio < DEAD),
            self.pick(ratio > NOISY),
            self.pick((np.abs(correlation) * (1 - np.eye(len(reference)))).max(axis=1) >= COPY),
        )


@dataclass(frozen=True)
class Coherence:
    """Complex coherency of sensor pairs in `BANDS` from 2 s Hann segments of a `rate` Hz recording."""

    rate: float
    positions: Float[np.ndarray, "c 3"]

    def spectra(self, signal: Float[np.ndarray, "c t"]) -> Complex[np.ndarray, "c s f"]:
        """Each channel's windowed, de-meaned segments' Fourier coefficients."""
        count = signal.shape[1] // SEGMENT
        cut = signal[:, : count * SEGMENT].astype(np.float32).reshape(len(signal), count, SEGMENT)
        return np.fft.rfft((cut - cut.mean(axis=2, keepdims=True)) * np.hanning(SEGMENT), axis=2)

    def masks(self) -> Bool[np.ndarray, "b f"]:
        """Which frequency bins of a segment each band holds."""
        frequencies = np.fft.rfftfreq(SEGMENT, 1.0 / self.rate)
        return np.array([(frequencies >= low) & (frequencies < high) for low, high in BANDS])

    def coherency(
        self, spectra: Complex[np.ndarray, "c s f"], pairs: Int[np.ndarray, "p 2"]
    ) -> Complex[np.ndarray, "p b"]:
        """The band-pooled coherency `S_xy / sqrt(S_xx S_yy)` of each pair, `x` the first sensor."""
        masks = self.masks()
        out = np.empty((len(pairs), len(masks)), dtype=np.complex64)
        for start in range(0, len(pairs), CHUNK):
            chunk = pairs[start : start + CHUNK]
            x, y = spectra[chunk[:, 0]], spectra[chunk[:, 1]]
            cross, power_x, power_y = x * y.conj(), np.abs(x) ** 2, np.abs(y) ** 2
            for band, mask in enumerate(masks):
                scale = np.sqrt(power_x[..., mask].sum((1, 2)) * power_y[..., mask].sum((1, 2)))
                with np.errstate(invalid="ignore", divide="ignore"):
                    out[start : start + CHUNK, band] = cross[..., mask].sum((1, 2)) / scale
        return out

    def distances(self, pairs: Int[np.ndarray, "p 2"]) -> Float[np.ndarray, " p"]:
        """The metres between the two sensors of each pair."""
        return np.linalg.norm(self.positions[pairs[:, 0]] - self.positions[pairs[:, 1]], axis=1)

    @staticmethod
    def draw(channels: int, count: int, seed: int) -> Int[np.ndarray, "p 2"]:
        """`count` sensor pairs, uniformly, no sensor with itself."""
        rng = np.random.default_rng(seed)
        first = rng.integers(0, channels, size=count)
        second = (first + rng.integers(1, channels, size=count)) % channels
        return np.stack([first, second], axis=1)


@dataclass(frozen=True)
class Coherencies:
    """Each sensor pair's band-pooled complex coherency."""

    values: Complex[np.ndarray, "p b"]

    def medians(self) -> Float[np.ndarray, "b 2"]:
        """Per band the median |real| and median |imag| part."""
        return np.stack(
            [np.median(np.abs(self.values.real), axis=0), np.median(np.abs(self.values.imag), axis=0)], axis=1
        )

    def by_distance(self, distance: Float[np.ndarray, " p"]) -> Float[np.ndarray, "b d"]:
        """Per band the median |coherency| of the pairs in each distance bin (nan for an empty bin)."""
        bins = np.digitize(distance, DISTANCES[1:-1])
        table = np.full((self.values.shape[1], len(DISTANCES) - 1), np.nan)
        for index in range(table.shape[1]):
            held = bins == index
            if held.any():
                table[:, index] = np.median(np.abs(self.values[held]), axis=0)
        return table


@dataclass(frozen=True)
class Selection:
    """The training runs of subject 0 the study reads."""

    runs: list[Run]

    @classmethod
    def of(cls, runs: list[Run], count: int = SESSIONS) -> "Selection":
        """`count` of subject 0's training runs, evenly spread through their sorted order."""
        trained = sorted(
            (run for run in runs if run.subject == DEEP and run.partition is Partition.TRAIN),
            key=lambda run: run.name,
        )
        picks = np.unique(np.linspace(0, len(trained) - 1, num=count).round().astype(int))
        return cls([trained[i] for i in picks])

    @staticmethod
    def central(signal: Float[np.ndarray, "c t"], rate: float, minutes: float = MINUTES) -> Float[np.ndarray, "c m"]:
        """The middle `minutes` of `signal`, all of it when it is shorter."""
        width = int(minutes * SECONDS * rate)
        begin = max(0, (signal.shape[1] - width) // 2)
        return np.array(signal[:, begin : begin + width])


@dataclass(frozen=True)
class Study:
    """The readouts over a selection of cached runs."""

    cache: RunCache
    sensors: Sensors
    pairs: Int[np.ndarray, "p 2"]
    minutes: float

    def session(self, run: Run) -> Complex[np.ndarray, "p b"]:
        """Log one run's health and per-band medians; return its pair coherencies."""
        signal = np.load(self.cache.path(run.name), mmap_mode="r")
        central = Selection.central(signal, self.cache.rate, self.minutes)
        health = self.sensors.health(np.asarray(signal), central)
        coherence = Coherence(self.cache.rate, self.sensors.positions)
        coherency = coherence.coherency(coherence.spectra(central), self.pairs)
        minutes = central.shape[1] / self.cache.rate / SECONDS
        LOG.info(
            "%s %s read %.1f of %.1f min",
            run.name,
            health.summary(),
            minutes,
            signal.shape[1] / self.cache.rate / SECONDS,
        )
        LOG.info("%s flagged %s", run.name, health.listing())
        self.medians(run.name, coherency)
        return coherency

    @staticmethod
    def medians(label: str, coherency: Complex[np.ndarray, "p b"]) -> None:
        for band, (real, imag) in zip(BANDS, Coherencies(coherency).medians(), strict=True):
            LOG.info("%s band %g-%g Hz median|real| %.3f median|imag| %.3f", label, *band, real, imag)

    def pooled(self, runs: list[Run]) -> None:
        """Every run's readouts, then |coherency| by distance over all of them."""
        coherencies = [self.session(run) for run in runs]
        stacked = np.concatenate(coherencies)
        distance = np.tile(Coherence(self.cache.rate, self.sensors.positions).distances(self.pairs), len(runs))
        counts = np.bincount(np.digitize(distance, DISTANCES[1:-1]), minlength=len(DISTANCES) - 1)
        self.medians(f"pooled({len(runs)})", stacked)
        LOG.info("median |coherency| by distance bins %s m, pairs per bin %s", DISTANCES, counts)
        for band, row in zip(BANDS, Coherencies(stacked).by_distance(distance), strict=True):
            LOG.info("band %g-%g Hz %s", *band, np.round(row, 3))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", type=int, default=SESSIONS)
    parser.add_argument("--minutes", type=float, default=MINUTES)
    parser.add_argument("--pairs", type=int, default=PAIRS)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    root = DataRoot.from_config()
    cache = RunCache.under(root.processed())
    sensors = Sensors.of(Neuromag.of(NEUROMAG.home(root)))
    pairs = Coherence.draw(len(sensors.names), args.pairs, args.seed)
    LOG.info("rate %g Hz, bands %s, %d pairs, segments %d samples", cache.rate, BANDS, len(pairs), SEGMENT)
    runs = Selection.of(Run.discover(root.raw()), args.sessions).runs
    Study(cache, sensors, pairs, args.minutes).pooled(runs)


if __name__ == "__main__":
    main()
