"""A sample's MEG and stimulus audio, cut to one window on the MEG clock and written as the viewer reads them."""

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
from jaxtyping import Float, Int, Shaped
from scipy.io import wavfile

from megspeech.data.raw.events import Event, Events
from megspeech.data.raw.runs import Run
from megspeech.objective.targets.speech import LAG, Heard

PCM = np.iinfo(np.int16).max


@dataclass(frozen=True)
class Meg:
    """One run's signal, `[channels, samples]` at `rate`."""

    signal: Float[np.ndarray, "c t"]
    rate: float

    def window(self, start: int, length: int) -> Float[np.ndarray, "c s"]:
        """`length` samples from `start`, zero where they fall outside the run."""
        out = np.zeros((self.signal.shape[0], length), dtype=np.float32)
        low, high = max(start, 0), min(start + length, self.signal.shape[1])
        if high > low:
            out[:, low - start : high - start] = self.signal[:, low:high]
        return out

    @staticmethod
    def zscored(window: Float[np.ndarray, "c s"]) -> Float[np.ndarray, "c s"]:
        """Each channel centred and scaled to unit deviation; a flat channel stays zero."""
        centred = window - window.mean(axis=1, keepdims=True)
        deviation = centred.std(axis=1, keepdims=True)
        return (centred / np.where(deviation > 0, deviation, 1)).astype(np.float32)

    @staticmethod
    def write(path: Path, window: Float[np.ndarray, "c s"]) -> Path:
        """`window` as little-endian float32, row-major `[channels, samples]`."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(np.ascontiguousarray(window, dtype="<f4").tobytes())
        return path


@dataclass(frozen=True)
class Stimulus:
    """A run's chapter audio and the word onsets that place it on the MEG clock; empty where none ships."""

    onsets: Float[np.ndarray, " n"]
    chapter: Float[np.ndarray, " n"]
    path: Path | None

    @classmethod
    def of(cls, run: Run) -> "Stimulus":
        chapters = Heard(run).chapters()
        if not chapters:
            return cls(np.zeros(0), np.zeros(0), None)
        heard = Events(run.events).words(Event.ONSET, Event.CHAPTER).dropna(subset=[Event.ONSET, Event.CHAPTER])
        return cls(
            heard[Event.ONSET].to_numpy(dtype=np.float64),
            heard[Event.CHAPTER].to_numpy(dtype=np.float64) + LAG,
            chapters[0],
        )

    @cached_property
    def sound(self) -> tuple[int, Int[np.ndarray, " a"]]:
        """The chapter's rate and its mono 16-bit samples."""
        if self.path is None:
            return 0, np.zeros(0, dtype=np.int16)
        rate, samples = wavfile.read(self.path)
        return rate, self.pcm(samples)

    @staticmethod
    def pcm(samples: Shaped[np.ndarray, "a *c"]) -> Int[np.ndarray, " a"]:
        """Any wav's samples as mono 16-bit: channels averaged, floats scaled from [-1, 1]."""
        mono = samples.mean(axis=1) if samples.ndim > 1 else samples
        if np.issubdtype(samples.dtype, np.floating):
            mono = mono * PCM
        return np.clip(np.round(mono), -PCM - 1, PCM).astype(np.int16)

    def at(self, meg: float) -> float:
        """The chapter time of MEG time `meg`, through the nearest heard word's onset."""
        nearest = int(np.abs(self.onsets - meg).argmin())
        return float(self.chapter[nearest] + meg - self.onsets[nearest])

    def clip(self, t0: float, seconds: float) -> tuple[int, Int[np.ndarray, " a"]] | None:
        """The audio heard from MEG time `t0` for `seconds`, zero past the chapter's ends; None without audio."""
        if self.path is None or not len(self.onsets):
            return None
        rate, sound = self.sound
        start, length = round(self.at(t0) * rate), round(seconds * rate)
        out = np.zeros(length, dtype=np.int16)
        low, high = max(start, 0), min(start + length, len(sound))
        if high > low:
            out[low - start : high - start] = sound[low:high]
        return rate, out

    @staticmethod
    def write(path: Path, rate: int, audio: Int[np.ndarray, " a"]) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        wavfile.write(path, rate, audio.astype(np.int16))
        return path
