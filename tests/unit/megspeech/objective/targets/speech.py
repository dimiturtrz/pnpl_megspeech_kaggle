import logging
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.io import wavfile
from torch import Tensor

from core.paths import DataRoot
from megspeech.data.raw.runs import Run
from megspeech.objective.targets import speech as module
from megspeech.objective.targets.speech import (
    BLOCK,
    CHUNK,
    LAG,
    SAMPLING,
    STEM,
    STRIDE,
    WINDOW,
    Chapter,
    Heard,
    Reading,
    Speech,
    Targets,
    Teacher,
)
from megspeech.objective.targets.t5 import Encoder as T5Encoder
from tests.unit.megspeech.data.config import HELD, TRAIN

DIM = 2
LAYERS = 25
CPU = torch.device("cpu")
SECONDS = 4


@dataclass(frozen=True)
class Encoded:
    hidden_states: tuple[Tensor, ...]


class Encoder:
    """Every layer's frame is the sample it starts at, so a latent names the audio it was read from."""

    def __call__(self, *, input_values: Tensor, output_hidden_states: bool) -> Encoded:
        frames = (input_values.shape[1] - 400) // STRIDE + 1
        starts = input_values[:, ::STRIDE][:, :frames, None].expand(-1, -1, DIM)
        return Encoded((starts,) * LAYERS)

    def to(self, _: torch.device) -> "Encoder":
        return self

    def eval(self) -> "Encoder":
        return self


class Loader:
    def __init__(self) -> None:
        self.seen: list[object] = []

    def from_pretrained(self, *args: object, **__: object) -> Encoder:
        self.seen.extend(args)
        return Encoder()


class Root:
    def __init__(self, path: Path) -> None:
        self.base = path

    def raw(self) -> Path:
        return self.base / "raw"

    def processed(self) -> Path:
        return self.base / "processed"

    def foreign(self, name: str) -> Path:
        return self.base / name


def encoder() -> T5Encoder:
    return cast("T5Encoder", Encoder())


def loud(seconds: float, at: float) -> Chapter:
    """Silence until `at`, a 440 Hz tone after."""
    time = np.arange(round(seconds * SAMPLING)) / SAMPLING
    return Chapter((np.sin(2 * np.pi * 440 * time) * (time >= at)).astype(np.float32))


def heard_run(root: Path, name: str = TRAIN, *, audio: bool = True) -> Run:
    """A run under `root/raw` whose events heard three words, the second outside any segment, and whose session's
    chapter, `SECONDS` long, is silent until 1 s."""
    task = root / "raw" / "Sherlock1"
    run = Run(task / "derivatives" / "serialised" / name)
    run.meg.parent.mkdir(parents=True, exist_ok=True)
    run.meg.touch()
    run.events.parent.mkdir(parents=True, exist_ok=True)
    events = pd.DataFrame(
        {
            "kind": ["word", "phoneme", "word", "word"],
            "segment": ["a", "a", "", "b"],
            "timechapter": [0.94, 0.95, 2.0, 3.8],
            "timemeg": [10.94, 10.95, 12.0, 13.8],
            "sentenceidx": [0, 0, 0, 0],
            "duration": [0.2, 0.01, 0.2, 0.2],
        }
    )
    events.to_csv(run.events, sep="\t", index=False)
    if audio:
        (task / "stimuli" / "audio").mkdir(parents=True, exist_ok=True)
        _, session, _ = run.key
        wav = task / "stimuli" / "audio" / f"sherlock1_{session:02d}_doyle_64kb.wav"
        wavfile.write(wav, SAMPLING, (loud(SECONDS, 1.0).audio * 1000).astype(np.int16))
    return run


def test_read(tmp_path: Path) -> None:
    path = tmp_path / "chapter.wav"
    stereo = np.stack([np.arange(8000), -np.arange(8000) // 2], axis=1).astype(np.int16)
    wavfile.write(path, SAMPLING // 2, stereo)
    audio = Chapter.read(path).audio
    assert len(audio) == 2 * 8000
    assert audio.mean() == pytest.approx(0.0, abs=1e-5)
    assert audio.std() == pytest.approx(1.0, rel=1e-4)


def test_frames() -> None:
    assert Chapter(np.zeros(400, np.float32)).frames == 1
    assert Chapter(np.zeros(399, np.float32)).frames == 0
    assert Chapter(np.zeros(SAMPLING, np.float32)).frames == SAMPLING // STRIDE - 1


def test_latents() -> None:
    """Chunks are stitched in order, every frame read once from its own place in the chapter."""
    chapter = Chapter((np.arange(round(2.5 * CHUNK * STRIDE)) % 997).astype(np.float32))
    latents = chapter.latents(encoder(), CPU)
    assert latents.shape == (chapter.frames, DIM)
    expected = torch.from_numpy(chapter.audio[::STRIDE][: chapter.frames])
    assert torch.equal(latents[:, 0], expected)


def test_contrast() -> None:
    contrast = loud(2.0, 1.0).contrast(np.array([1.0, 0.5, 0.01, 1.98, np.nan]))
    assert contrast[0] > np.log(100)
    assert contrast[1] == pytest.approx(0.0)
    assert np.isnan(contrast[2:]).all()


def test_pool() -> None:
    latents = torch.arange(100.0)[:, None].expand(-1, DIM)
    pooled = Targets.pool(latents, np.array([0.0, 1.0, 1.6, -0.1, np.nan]))
    assert pooled.vectors.dtype == np.float16
    assert pooled.vectors[:2, 0].tolist() == [(WINDOW - 1) / 2, 50 + (WINDOW - 1) / 2]
    assert np.isnan(pooled.vectors[2:]).all()


def test_aligned() -> None:
    contrast = np.concatenate([np.full(BLOCK, 1.0), np.full(BLOCK, 0.2), np.full(BLOCK // 2, np.nan)])
    kept = Targets.aligned(contrast)
    assert kept[:BLOCK].all()
    assert not kept[BLOCK:].any()


def test_masked() -> None:
    masked = Targets(np.ones((3, DIM), np.float16)).masked(np.array([True, False, True]))
    assert masked.present.tolist() == [True, False, True]


def test_present() -> None:
    vectors = np.array([[1.0, 2.0], [np.nan, 2.0]], np.float16)
    assert Targets(vectors).present.tolist() == [True, False]


def test_directory(tmp_path: Path) -> None:
    assert Targets.directory(cast("DataRoot", Root(tmp_path)), "gpt2-large").name == "gpt2-large"
    assert Targets.directory(cast("DataRoot", Root(tmp_path))) == tmp_path / "processed" / "speech" / STEM


def test_save(tmp_path: Path) -> None:
    path = Targets(np.ones((2, DIM), np.float16)).save(tmp_path / "nested" / "run.npy")
    assert np.load(path).shape == (2, DIM)


def test_chapters(tmp_path: Path) -> None:
    assert [path.name for path in Heard(heard_run(tmp_path)).chapters()] == ["sherlock1_01_doyle_64kb.wav"]
    assert Heard(heard_run(tmp_path, HELD, audio=False)).chapters() == []


def test_onsets(tmp_path: Path) -> None:
    assert Heard(heard_run(tmp_path)).onsets() == pytest.approx([0.94 + LAG, 3.8 + LAG])


def test_onsets_align_with_run_words(tmp_path: Path) -> None:
    """Row i of a run's targets is word i of `Run.words`, the positional index `Speech.gather` reads them by."""
    run = heard_run(tmp_path)
    assert Heard(run).onsets() + 10 - LAG == pytest.approx(run.words()["onset"].to_numpy())


def test_targets(tmp_path: Path) -> None:
    """The first word starts the tone, so its block is aligned; the second's 0.5 s run past the chapter's end."""
    targets = Heard(heard_run(tmp_path)).targets(encoder(), CPU)
    assert targets.vectors.shape == (2, DIM)
    assert targets.present.tolist() == [True, False]


def test_trained(tmp_path: Path) -> None:
    train, held = heard_run(tmp_path), heard_run(tmp_path, HELD)
    assert [entry.run for entry in Heard.trained([train, held])] == [train]
    assert Heard.trained([heard_run(tmp_path / "silent", audio=False)]) == []


def test_write(tmp_path: Path) -> None:
    heard_run(tmp_path)
    heard_run(tmp_path, HELD)
    root = cast("DataRoot", Root(tmp_path))
    directory = Heard.write(root, encoder(), CPU, Reading())
    assert [path.stem for path in directory.glob("*.npy")] == [Run(Path(TRAIN)).name]
    assert Heard.write(root, encoder(), CPU, Reading()) == directory
    hubert = Heard.write(root, encoder(), CPU, Reading(Teacher.HUBERT))
    assert hubert == tmp_path / "processed" / "speech" / "hubert-large-ll60k"
    assert [path.stem for path in hubert.glob("*.npy")] == [Run(Path(TRAIN)).name]


def test_stem() -> None:
    """Each teacher and layer span has its own cache; the default wav2vec2 one keeps its historical name."""
    assert Reading().stem == STEM
    assert Reading(Teacher.HUBERT).stem == "hubert-large-ll60k"
    assert Reading(Teacher.HUBERT, slice(12, 19)).stem == "hubert-large-ll60k-L12-18"


def test_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Each teacher loads through its own model class from its own folder."""
    loader = Loader()
    monkeypatch.setitem(module.LOADERS, Teacher.HUBERT, loader)
    root = cast("DataRoot", Root(tmp_path))
    assert isinstance(Reading(Teacher.HUBERT).load(root, CPU), Encoder)
    assert loader.seen == [tmp_path / "hubert-large-ll60k"]


def test_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    heard_run(tmp_path)
    monkeypatch.setitem(module.LOADERS, Teacher.WAV2VEC2, Loader())
    monkeypatch.setitem(module.LOADERS, Teacher.HUBERT, Loader())
    monkeypatch.setattr(module, "DEVICE", "cpu")
    monkeypatch.setattr(module.DataRoot, "from_config", classmethod(lambda _: Root(tmp_path)))
    monkeypatch.setattr("sys.argv", ["speech"])
    Heard.main()
    assert np.load(tmp_path / "processed" / "speech" / STEM / f"{Run(Path(TRAIN)).name}.npy").shape == (2, DIM)
    monkeypatch.setattr("sys.argv", ["speech", "--teacher", "hubert-large-ll60k", "--layers", "12", "18"])
    Heard.main()
    assert (tmp_path / "processed" / "speech" / "hubert-large-ll60k-L12-18" / f"{Run(Path(TRAIN)).name}.npy").exists()


def test_gather(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    np.save(tmp_path / "a.npy", np.array([[1, 2], [np.nan, np.nan], [3, 4]], np.float16))
    words = pd.DataFrame({"run": ["a", "a", "b", "a", "a"], "position": [2, 0, 0, 1, 0]})
    with caplog.at_level(logging.INFO):
        speech = Speech.gather(tmp_path, words)
    assert speech.rows.tolist() == [0, 1, -1, -1, 1]
    assert speech.vectors.tolist() == [[3, 4], [1, 2], [1, 2]]
    assert "3 of 5 words, 1 of 2 runs" in caplog.text
    assert Speech.gather(tmp_path, words, centred=True).vectors.tolist() == [[1, 1], [-1, -1], [-1, -1]]
    with pytest.raises(FileNotFoundError):
        Speech.gather(tmp_path / "absent", words)
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        Speech.gather(tmp_path / "empty", words)


def test_centred() -> None:
    vectors = np.array([[1.0, 2.0], [np.nan, np.nan], [3.0, 6.0]], np.float16)
    centred = Targets(vectors).centred()
    assert centred.vectors[[0, 2]].tolist() == [[-1, -2], [1, 2]]
    assert centred.present.tolist() == [True, False, True]
    assert not Targets(np.full((1, DIM), np.nan, np.float16)).centred().present.any()


def test_merged() -> None:
    """Words with an identical target share a row; distinct targets and absent words keep theirs."""
    speech = Speech(torch.tensor([0, 1, 2, -1, 3]), torch.tensor([[1.0, 2], [3, 4], [1, 2], [5, 6]]))
    assert speech.merged().rows.tolist() == [0, 1, 0, -1, 3]
    empty = Speech(torch.tensor([-1]), torch.zeros(0, DIM))
    assert empty.merged().rows.tolist() == [-1]


def test_to() -> None:
    speech = Speech(torch.tensor([0, -1]), torch.ones(1, DIM)).to(CPU)
    assert speech.rows.device == CPU
    assert speech.vectors.device == CPU


def test_standardised() -> None:
    vectors = np.array([[1.0, 5.0], [np.nan, np.nan], [3.0, 5.0]], np.float16)
    standard = Targets(vectors).standardised()
    assert standard.vectors[[0, 2], 0].tolist() == [-1, 1]
    assert standard.vectors[[0, 2], 1].tolist() == [0, 0]
    assert standard.present.tolist() == [True, False, True]
    empty = Targets(np.full((1, DIM), np.nan, np.float16))
    assert empty.standardised() is empty
