"""Frozen self-supervised speech latents (a `Teacher`: wav2vec2, HuBERT or data2vec) of the audiobook speech each
training word was heard in: token-level acoustic targets (Défossez et al. 2023), where the t5 lexicon is one vector per
word type.

Each trained Sherlock run's chapter (`<book>_<session>_doyle_64kb.wav`) is encoded once; a word's target is the
mean of the latents over the `WINDOW` after its chapter onset, a row of `<run>.npy` aligned with the run's word
table (`Run.words`), NaN where there is none. Validation and test runs' audio is never read.

uv run python -m megspeech.objective.targets.speech [--teacher hubert-large-ll60k] [--layers 15 18]
"""

import argparse
import logging
from dataclasses import dataclass
from enum import StrEnum
from math import gcd
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from jaxtyping import Bool, Float, Int
from scipy.io import wavfile
from scipy.signal import resample_poly
from torch import Tensor
from transformers import Data2VecAudioModel, HubertModel, Wav2Vec2Model

from core.paths import DataRoot
from megspeech.data.config import DataConfig, Val
from megspeech.data.raw.events import Event, Events
from megspeech.data.raw.runs import Column, Partition, Run
from megspeech.device import Device
from megspeech.objective.targets.t5 import Encoder


class Teacher(StrEnum):
    """The frozen speech model whose latents are the targets, named by its weights' folder under the foreign models.
    All are 24-layer, 1024-wide, 16 kHz models on the same convolutional front end (`STRIDE`, `RECEPTIVE`), fed
    normalised audio, so one pipeline reads either."""

    WAV2VEC2 = "wav2vec2-large-xlsr-53"
    HUBERT = "hubert-large-ll60k"
    """Libri-Light's 60k h of LibriVox audiobooks, LibriBrain's own domain, where xlsr-53 spreads over 53 languages;
    HuBERT-class models have the strongest brain-encoding record of the frame-level SSL models (Vaidya et al. 2022,
    Li et al. 2023; research/2026-10-07_target_encoders.md)."""
    DATA2VEC = "data2vec-audio-large"
    """LibriSpeech's 960 h, regressing the teacher's own averaged top layers rather than HuBERT's discrete units: a
    third objective on the same front end, for an ensemble whose same-teacher members saturate."""


LOADERS = {Teacher.WAV2VEC2: Wav2Vec2Model, Teacher.HUBERT: HubertModel, Teacher.DATA2VEC: Data2VecAudioModel}
STEM = Teacher.WAV2VEC2.value
LAYERS = slice(15, 19)
"""Transformer layers 15-18 of the 24, averaged: Défossez et al. (2023) align MEG to mid-to-late wav2vec2 layers, where
phonetic and lexical content peaks. HuBERT's default too, with no alignment study of its own yet (7gf.137): the
brain-encoding studies put the best speech-model layers in the upper middle (Vaidya et al. 2022), the shortlist's
12-18, and one window keeps a teacher swap a one-variable change; `--layers` reads another."""
SAMPLING = 16_000
"""The rate both teachers were pretrained at."""
STRIDE = 320
"""Samples per latent frame: 50 frames a second at `SAMPLING`."""
RECEPTIVE = 400
"""Samples the convolutional encoder reads for one frame."""
CHUNK = 30 * SAMPLING // STRIDE
"""Frames kept from one forward pass: 30 s, far below the quadratic-attention memory ceiling."""
CONTEXT = 5 * SAMPLING // STRIDE
"""Frames of audio read on each side of a chunk and discarded, so every kept frame attends to >= 5 s around it."""
WINDOW = SAMPLING // 2 // STRIDE
"""Frames a word's target averages: the 0.5 s after its onset, the acoustic content of the MEG window's response."""
LAG = 0.06
"""Seconds the audio runs behind the events' chapter clock: word-onset energy contrast peaks at +0.05-0.07 s on every
aligned run."""
ONSET = SAMPLING // 20
"""Samples (50 ms) whose energy after a word's onset, against the 50 ms ending 50 ms before it, is its contrast."""
BLOCK = 200
"""Consecutive words whose targets are kept or dropped together, by their mean contrast."""
ALIGNED = 0.7
"""Least mean log energy contrast of an aligned block: the 2821 blocks of the 99 trained runs are bimodal, those of a
different recording (Sherlock3 sessions 1-8, Sherlock4 8, Sherlock6 14) or past a splice at -0.7-0.6, the rest at
0.8-2.5 bar 3 boundary blocks; the middle of the trough between."""
SILENCE = 1e-3
"""Energy floor, as a fraction of the chapter's mean energy, so a silent stretch has a finite contrast."""
FLOOR = 1e-6
"""Least per-dimension spread `Targets.standardised` divides by."""
KEY_SEED = 0
"""Seed of the random direction a target is keyed by to tell identical targets apart."""
DEVICE = Device.CUDA
LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class Reading:
    """A teacher and the span of its layers whose mean hidden state is the target."""

    teacher: Teacher = Teacher.WAV2VEC2
    layers: slice = LAYERS

    @property
    def stem(self) -> str:
        """The targets' directory under processed/speech: the teacher's name, suffixed by the layers unless `LAYERS`,
        so every teacher and layer span has its own cache and the wav2vec2 one keeps its name."""
        suffix = "" if self.layers == LAYERS else f"-L{self.layers.start}-{self.layers.stop - 1}"
        return f"{self.teacher}{suffix}"

    def load(self, root: DataRoot, device: torch.device) -> Encoder:
        return LOADERS[self.teacher].from_pretrained(root.foreign(self.teacher.value)).to(device).eval()


@dataclass(frozen=True)
class Chapter:
    """A chapter's audio at `SAMPLING`, zero-mean and unit-variance as wav2vec2's feature extractor feeds it."""

    audio: Float[np.ndarray, " s"]

    @classmethod
    def read(cls, path: Path) -> "Chapter":
        rate, samples = wavfile.read(path)
        mono = samples.reshape(len(samples), -1).astype(np.float64).mean(axis=1)
        common = gcd(SAMPLING, rate)
        audio = resample_poly(mono, SAMPLING // common, rate // common)
        return cls(((audio - audio.mean()) / audio.std()).astype(np.float32))

    @property
    def frames(self) -> int:
        return max(0, (len(self.audio) - RECEPTIVE) // STRIDE + 1)

    def latents(self, encoder: Encoder, device: torch.device, layers: slice = LAYERS) -> Float[Tensor, "f d"]:
        """Every frame's mean `layers` hidden state, chunk by chunk, each chunk read with `CONTEXT` on both sides."""
        parts: list[Tensor] = []
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16):
            for start in range(0, self.frames, CHUNK):
                first = max(0, start - CONTEXT)
                stop = min(self.frames, start + CHUNK)
                end = min(self.frames, stop + CONTEXT)
                audio = torch.from_numpy(self.audio[first * STRIDE : (end - 1) * STRIDE + RECEPTIVE]).to(device)
                hidden = encoder(input_values=audio[None], output_hidden_states=True).hidden_states[layers]
                parts.append(torch.stack(hidden).float().mean(0)[0, start - first : stop - first])
        return torch.cat(parts)

    def contrast(self, onset: Float[np.ndarray, " n"]) -> Float[np.ndarray, " n"]:
        """Each onset's log energy over the `ONSET` after it against the `ONSET` ending `ONSET` before it; NaN where
        either falls outside the audio."""
        energy = np.concatenate([[0.0], np.cumsum(self.audio.astype(np.float64) ** 2)])
        floor = SILENCE * energy[-1] / max(len(self.audio), 1)
        at = np.round(np.nan_to_num(onset, nan=-1.0) * SAMPLING).astype(np.int64)
        inside = np.isfinite(onset) & (at >= 2 * ONSET) & (at + ONSET <= len(self.audio))
        at = np.where(inside, at, 2 * ONSET)
        after = (energy[at + ONSET] - energy[at]) / ONSET
        before = (energy[at - ONSET] - energy[at - 2 * ONSET]) / ONSET
        return np.where(inside, np.log((after + floor) / (before + floor)), np.nan)


@dataclass(frozen=True)
class Targets:
    """One run's per-word speech targets, NaN rows where the word has none."""

    vectors: Float[np.ndarray, "n d"]

    @classmethod
    def pool(cls, latents: Float[Tensor, "f d"], onset: Float[np.ndarray, " n"]) -> "Targets":
        """Each word's mean latent over the `WINDOW` frames from its onset; NaN past either end of the audio."""
        first = np.floor(np.nan_to_num(onset, nan=-1.0) * SAMPLING / STRIDE).astype(np.int64)
        inside = np.isfinite(onset) & (first >= 0) & (first + WINDOW <= len(latents))
        start = torch.from_numpy(np.where(inside, first, 0)).to(latents.device)
        summed = torch.cat([latents.new_zeros(1, latents.shape[1]), latents.double().cumsum(0)])
        mean = ((summed[start + WINDOW] - summed[start]) / WINDOW).float().cpu().numpy()
        return cls(np.where(inside[:, None], mean, np.nan).astype(np.float16))

    @staticmethod
    def aligned(contrast: Float[np.ndarray, " n"]) -> Bool[np.ndarray, " n"]:
        """Words in a `BLOCK` whose mean contrast reaches `ALIGNED`: the audio there is the recording the events
        were timed on."""
        blocks = np.arange(len(contrast)) // BLOCK
        mean = pd.Series(contrast).groupby(blocks).transform("mean").to_numpy()
        return np.nan_to_num(mean, nan=-np.inf) >= ALIGNED

    def masked(self, keep: Bool[np.ndarray, " n"]) -> "Targets":
        return Targets(np.where(keep[:, None], self.vectors, np.float16(np.nan)))

    def centred(self) -> "Targets":
        """Less the run's mean target over its present words: the direction all targets share removed."""
        mean = np.nanmean(self.vectors.astype(np.float32), axis=0) if self.present.any() else 0.0
        return Targets((self.vectors - mean).astype(np.float16))

    def standardised(self) -> "Targets":
        """Each dimension's mean removed and spread divided out over the run's present words."""
        if not self.present.any():
            return self
        vectors = self.vectors.astype(np.float32)
        spread = np.nanstd(vectors, axis=0)
        return Targets(((vectors - np.nanmean(vectors, axis=0)) / np.maximum(spread, FLOOR)).astype(np.float16))

    @property
    def present(self) -> Bool[np.ndarray, " n"]:
        return np.isfinite(self.vectors).all(axis=1)

    @staticmethod
    def directory(root: DataRoot, source: str = "") -> Path:
        """Where the per-word targets of `source` (the wav2vec2 latents when empty; a `Reading.stem`) live."""
        return root.processed() / "speech" / (source or STEM)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self.vectors)
        return path


@dataclass(frozen=True)
class Heard:
    """A training run and the chapter it played."""

    run: Run

    def chapters(self) -> list[Path]:
        """The audio of the run's session in its task's stimuli: one chapter, or none where the dataset ships none."""
        _, session, _ = self.run.key
        return sorted((self.run.meg.parents[2] / "stimuli" / "audio").glob(f"*_{session:02d}_*.wav"))

    def onsets(self) -> Float[np.ndarray, " n"]:
        """Each heard word's (`Run.words`) onset in its chapter's audio."""
        heard = Events(self.run.events).words(Event.CHAPTER)
        return heard[Event.CHAPTER].to_numpy(dtype=np.float64) + LAG

    def targets(self, encoder: Encoder, device: torch.device, layers: slice = LAYERS) -> Targets:
        chapter = Chapter.read(self.chapters()[0])
        onset = self.onsets()
        aligned = Targets.aligned(chapter.contrast(onset))
        return Targets.pool(chapter.latents(encoder, device, layers), onset).masked(aligned)

    @staticmethod
    def trained(runs: list[Run]) -> list["Heard"]:
        """The default config's training runs whose chapter audio is present: never a validation or test run."""
        heard = [Heard(run) for run in DataConfig(val=Val.OFFICIAL).trained(runs) if run.partition is Partition.TRAIN]
        return [entry for entry in heard if entry.chapters()]

    @staticmethod
    def write(root: DataRoot, encoder: Encoder, device: torch.device, reading: Reading) -> Path:
        directory = Targets.directory(root, reading.stem)
        for entry in Heard.trained(Run.discover(root.raw())):
            path = directory / f"{entry.run.name}.npy"
            if path.exists():
                continue
            targets = entry.targets(encoder, device, reading.layers)
            targets.save(path)
            LOG.info("%s: %d of %d words", entry.run.name, targets.present.sum(), len(targets.vectors))
        return directory

    @staticmethod
    def main() -> None:
        """Encode every trained run's chapter on the GPU with the chosen teacher and layers and write its words'
        targets under processed/speech/<`Reading.stem`>."""
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--teacher", type=Teacher, choices=list(Teacher), default=Teacher.WAV2VEC2)
        parser.add_argument(
            "--layers", type=int, nargs=2, default=(LAYERS.start, LAYERS.stop - 1), metavar=("FIRST", "LAST")
        )
        args = parser.parse_args()
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
        reading = Reading(args.teacher, slice(args.layers[0], args.layers[1] + 1))
        root, device = DataRoot.from_config(), torch.device(DEVICE)
        LOG.info("wrote %s", Heard.write(root, reading.load(root, device), device, reading))


@dataclass(frozen=True)
class Speech:
    """Each training word's row of `vectors`, or -1 where it has no speech target."""

    rows: Int[Tensor, " n"]
    vectors: Float[Tensor, "k d"]

    @classmethod
    def gather(cls, directory: Path, words: pd.DataFrame, *, centred: bool = False) -> "Speech":
        """The targets of `words` (`run`, `position` in its run's word table), in order, each run's `Targets.centred`
        when `centred`; a run without a file has none, and the coverage is logged."""
        first = next(directory.glob("*.npy"), None) if directory.is_dir() else None
        if first is None:
            raise FileNotFoundError(f"no targets under {directory}")
        words = words.reset_index(drop=True)
        vectors = np.full((len(words), np.load(first, mmap_mode="r").shape[1]), np.nan, dtype=np.float16)
        runs = words.groupby(Column.RUN, sort=False)
        for run, group in runs:
            path = directory / f"{run}.npy"
            if path.exists():
                targets = Targets(np.load(path))
                chosen = targets.centred() if centred else targets
                vectors[group.index] = chosen.vectors[group[Column.POSITION].to_numpy()]
        present = np.isfinite(vectors).all(axis=1)
        covered = sum((directory / f"{run}.npy").exists() for run in runs.groups)
        LOG.info("%s: %d of %d words, %d of %d runs", directory.name, present.sum(), len(words), covered, runs.ngroups)
        rows = np.where(present, np.cumsum(present) - 1, -1)
        return cls(torch.from_numpy(rows), torch.from_numpy(vectors[present])).merged()

    def merged(self) -> "Speech":
        """Words with an identical target pointing at one row, so the contrast counts them as one kind, not as
        negatives of each other."""
        if not len(self.vectors):
            return self
        keys = self.vectors.double() @ torch.randn(
            self.vectors.shape[1], dtype=torch.float64, generator=torch.Generator().manual_seed(KEY_SEED)
        )
        _, inverse = torch.unique(keys, return_inverse=True)
        first = torch.full((int(inverse.max()) + 1,), len(keys), dtype=torch.long)
        first = first.scatter_reduce(0, inverse, torch.arange(len(keys)), "amin")
        return Speech(torch.where(self.rows >= 0, first[inverse][self.rows.clamp(min=0)], self.rows), self.vectors)

    def to(self, device: torch.device) -> "Speech":
        return Speech(self.rows.to(device), self.vectors.to(device))


if __name__ == "__main__":
    Heard.main()
