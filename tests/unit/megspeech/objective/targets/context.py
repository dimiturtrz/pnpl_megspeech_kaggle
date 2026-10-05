from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest
import torch
from torch import Tensor

from core.paths import DataRoot
from megspeech.data.raw.runs import Run
from megspeech.objective.targets import context as module
from megspeech.objective.targets.context import LAYER, MODEL, Transcript
from megspeech.objective.targets.t5 import Encoder as T5Encoder
from tests.unit.megspeech.data.config import HELD, TRAIN

DIM = 2
CPU = torch.device("cpu")
SPACE = 32


@dataclass(frozen=True)
class Encoded:
    hidden_states: tuple[Tensor, ...]


class Encoder:
    """Layer k's state of a token is k+1 times the running sum of the ids in its window: it depends on the window."""

    def __call__(self, *, input_ids: Tensor, output_hidden_states: bool) -> Encoded:
        running = input_ids.cumsum(dim=1)[..., None].expand(-1, -1, DIM).float()
        return Encoded(tuple(running * (k + 1) for k in range(LAYER + 1)))

    def to(self, _: torch.device) -> "Encoder":
        return self

    def eval(self) -> "Encoder":
        return self


class Loader:
    def __init__(self, made: object) -> None:
        self.made = made

    def from_pretrained(self, *_: object, **__: object) -> object:
        return self.made


class Root:
    def __init__(self, path: Path) -> None:
        self.base = path

    def raw(self) -> Path:
        return self.base / "raw"

    def processed(self) -> Path:
        return self.base / "processed"

    def foreign(self, name: str) -> Path:
        return self.base / name


def tokenizer(texts: list[str]) -> dict[str, list[list[int]]]:
    """A character an id; a word that is only its space has no token."""
    return {"input_ids": [[ord(c) for c in text] if text.strip() else [] for text in texts]}


def encoder() -> T5Encoder:
    return cast("T5Encoder", Encoder())


def heard_run(root: Path, name: str = TRAIN) -> Run:
    """A run under `root/raw` that heard `ab` then `c`."""
    run = Run(root / "raw" / "Sherlock1" / "derivatives" / "serialised" / name)
    run.meg.parent.mkdir(parents=True, exist_ok=True)
    run.meg.touch()
    run.events.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "kind": ["word", "word"],
            "segment": ["ab", "c"],
            "sentenceidx": [0, 0],
            "timemeg": [0.5, 1.0],
            "duration": [0.2, 0.2],
        }
    ).to_csv(run.events, sep="\t", index=False)
    return run


def test_tokens() -> None:
    ids, owner = Transcript(("ab", "", "c")).tokens(tokenizer)
    assert ids.tolist() == [SPACE, ord("a"), ord("b"), SPACE, ord("c")]
    assert owner.tolist() == [0, 0, 0, 2, 2]


def test_states() -> None:
    """Windows of 4 advancing by 2, each later one keeping only its second half: the states restart per window."""
    ids = torch.arange(1, 7)
    states = Transcript.states(ids, encoder(), 1, 4)
    assert states[:, 0].tolist() == [2 * v for v in (1, 3, 6, 10, 12, 18)]
    assert Transcript.states(ids, encoder(), 1, 8)[:, 0].tolist() == [2 * v for v in (1, 3, 6, 10, 15, 21)]


def test_encode() -> None:
    """Each word its last token's state; the two present rows are z-scored to -1 and 1, the tokenless word is NaN."""
    targets = Transcript(("ab", "c", "")).encode(tokenizer, encoder(), CPU, layer=0, window=1024)
    assert targets.present.tolist() == [True, True, False]
    assert targets.vectors[:2].astype(np.float32) == pytest.approx(np.array([[-1.0] * DIM, [1.0] * DIM]), abs=1e-2)
    assert not Transcript(("",)).encode(tokenizer, encoder(), CPU).present.any()


def test_of(tmp_path: Path) -> None:
    assert Transcript.of(heard_run(tmp_path)).words == ("ab", "c")


def test_write(tmp_path: Path) -> None:
    heard_run(tmp_path)
    heard_run(tmp_path, HELD)
    root = cast("DataRoot", Root(tmp_path))
    directory = Transcript.write(root, tokenizer, encoder(), CPU)
    assert [path.stem for path in directory.glob("*.npy")] == [Run(Path(TRAIN)).name]
    assert Transcript.write(root, tokenizer, encoder(), CPU) == directory


def test_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    heard_run(tmp_path)
    monkeypatch.setattr(module, "AutoTokenizer", Loader(tokenizer))
    monkeypatch.setattr(module, "GPT2Model", Loader(Encoder()))
    monkeypatch.setattr(module, "DEVICE", "cpu")
    monkeypatch.setattr(module.DataRoot, "from_config", classmethod(lambda _: Root(tmp_path)))
    Transcript.main()
    assert np.load(tmp_path / "processed" / "speech" / MODEL / f"{Run(Path(TRAIN)).name}.npy").shape == (2, DIM)
