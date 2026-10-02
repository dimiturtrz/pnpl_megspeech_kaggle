from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from torch import Tensor

from megspeech.data.vocabulary import PRIMARY
from megspeech.objective import prototypes as module
from megspeech.objective.prototypes import LAYER, Prototypes

DIM = 4


class Tokenizer:
    def __call__(self, text: str, *, add_special_tokens: bool, return_tensors: str) -> dict[str, Tensor]:
        return {"input_ids": torch.tensor([[len(text), len(text) + 1]])}


@dataclass(frozen=True)
class Encoded:
    hidden_states: tuple[Tensor, ...]


class Encoder:
    def __call__(self, *, input_ids: Tensor, output_hidden_states: bool) -> Encoded:
        return Encoded(tuple(input_ids[..., None].float().expand(-1, -1, DIM) * (k + 1) for k in range(LAYER + 1)))

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

    def processed(self) -> Path:
        return self.base


def test_encode() -> None:
    encoded = Prototypes.encode(("ab", "abcd"), Tokenizer(), Encoder(), 1)
    assert encoded.vectors[:, 0].tolist() == [2 * 2.5, 2 * 4.5]


def test_path(tmp_path: Path) -> None:
    assert Prototypes.path(Root(tmp_path)).parent == tmp_path / "prototypes"  # type: ignore[arg-type]


def test_save(tmp_path: Path) -> None:
    assert Prototypes(torch.ones(2, DIM)).save(tmp_path / "p.npy").exists()


def test_load(tmp_path: Path) -> None:
    path = Prototypes(torch.ones(2, DIM)).save(tmp_path / "p.npy")
    assert torch.equal(Prototypes.load(path).vectors, torch.ones(2, DIM))


def test_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, "AutoTokenizer", Loader(Tokenizer()))
    monkeypatch.setattr(module, "T5EncoderModel", Loader(Encoder()))
    monkeypatch.setattr(module.DataRoot, "from_config", classmethod(lambda _: Root(tmp_path)))
    monkeypatch.setattr("sys.argv", ["prototypes"])
    Prototypes.main()
    assert Prototypes.load(Prototypes.path(Root(tmp_path))).vectors.shape == (len(PRIMARY), DIM)  # type: ignore[arg-type]
