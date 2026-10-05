from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
from torch import Tensor

from core.paths import DataRoot
from megspeech.data.raw.runs import Run
from megspeech.objective.targets import whisper as module
from megspeech.objective.targets.speech import STRIDE, Chapter, Heard
from megspeech.objective.targets.t5 import Encoder as T5Encoder
from megspeech.objective.targets.whisper import FRAMES, LAYERS, STEM, WINDOW, Listener
from tests.unit.megspeech.data.config import HELD, TRAIN
from tests.unit.megspeech.objective.targets.speech import CPU, DIM, Root, heard_run

STATES = 25
PEAK = 997


@dataclass(frozen=True)
class Encoded:
    hidden_states: tuple[Tensor, ...]


class Encoder:
    """Layer k's frame is k+1 times the audio sample it starts at, so a latent names where it was read from."""

    def __call__(self, *, input_features: Tensor, output_hidden_states: bool) -> Encoded:
        frames = input_features[:, 0, ::STRIDE][..., None].expand(-1, -1, DIM)
        return Encoded(tuple(frames * (k + 1) for k in range(STATES)))

    def to(self, _: torch.device) -> "Encoder":
        return self

    def eval(self) -> "Encoder":
        return self


def padded(windows: list[np.ndarray]) -> Tensor:
    """Each window zero-padded to `WINDOW` as raw samples, in place of a log-mel."""
    return torch.from_numpy(np.stack([np.pad(window, (0, WINDOW - len(window))) for window in windows]))[:, None, :]


class Features:
    def __init__(self, windows: list[np.ndarray]) -> None:
        self.input_features = padded(windows)


class FeatureLoader:
    def from_pretrained(self, *_: object) -> object:
        return lambda windows, **_: Features(windows)


class Model:
    encoder = Encoder()


class ModelLoader:
    def from_pretrained(self, *_: object) -> Model:
        return Model()


def listener() -> Listener:
    return Listener(padded, cast("T5Encoder", Encoder()))


def test_directory(tmp_path: Path) -> None:
    assert Listener.directory(cast("DataRoot", Root(tmp_path)), 12) == tmp_path / "processed" / "speech" / f"{STEM}-L12"


def test_latents() -> None:
    """Three overlapping windows are stitched in order, every frame read once from its own place in the chapter."""
    chapter = Chapter((1 + np.arange(round(65 * WINDOW / 30)) % PEAK).astype(np.float32))
    latents = listener().latents(chapter, CPU)
    assert set(latents) == set(LAYERS)
    assert FRAMES == WINDOW // STRIDE
    for layer, frames in latents.items():
        assert frames.shape == (chapter.frames, DIM)
        expected = (layer + 1) * torch.from_numpy(chapter.audio[::STRIDE][: chapter.frames]) / PEAK
        assert frames[:, 0].tolist() == pytest.approx(expected.tolist(), rel=1e-5)


def test_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, "WhisperFeatureExtractor", FeatureLoader())
    monkeypatch.setattr(module, "WhisperModel", ModelLoader())
    loaded = Listener.load(cast("DataRoot", Root(tmp_path)), CPU)
    assert loaded.extractor([np.zeros(8, np.float32)]).shape == (1, 1, WINDOW)


def test_targets(tmp_path: Path) -> None:
    """The run's first word is aligned and present at every layer, the second runs past the chapter's end."""
    targets = listener().targets(Heard(heard_run(tmp_path)), CPU)
    assert set(targets) == set(LAYERS)
    assert all(row.present.tolist() == [True, False] for row in targets.values())


def test_write(tmp_path: Path) -> None:
    heard_run(tmp_path)
    heard_run(tmp_path, HELD)
    root = cast("DataRoot", Root(tmp_path))
    written = listener().write(root, CPU)
    assert written == tmp_path / "processed" / "speech"
    for layer in LAYERS:
        assert [path.stem for path in Listener.directory(root, layer).glob("*.npy")] == [Run(Path(TRAIN)).name]
    assert listener().write(root, CPU) == written


def test_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    heard_run(tmp_path)
    monkeypatch.setattr(module, "WhisperFeatureExtractor", FeatureLoader())
    monkeypatch.setattr(module, "WhisperModel", ModelLoader())
    monkeypatch.setattr(module, "DEVICE", "cpu")
    monkeypatch.setattr(module.DataRoot, "from_config", classmethod(lambda _: Root(tmp_path)))
    Listener.main()
    root = cast("DataRoot", Root(tmp_path))
    assert np.load(Listener.directory(root, LAYERS[0]) / f"{Run(Path(TRAIN)).name}.npy").shape == (2, DIM)
