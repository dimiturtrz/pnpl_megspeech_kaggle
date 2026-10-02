from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import Tensor

from megspeech.data.continuous import DIRECTORY, RATE, WORDS
from megspeech.data.runs import CHANNELS
from megspeech.data.vocabulary import PRIMARY
from megspeech.objective import prototypes as module
from megspeech.objective.prototypes import LAYER, Lexicon, Prototypes

DIM = 4
CPU = torch.device("cpu")


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


def lexicon() -> Lexicon:
    return Lexicon(torch.tensor([-5, 3, 9]), torch.arange(3 * DIM).float().view(3, DIM))


def cache(root: Path) -> None:
    """One cached run that heard `The cat the zebra`, the last word too late for its window."""
    directory = root / DIRECTORY.format(rate=RATE)
    directory.mkdir(parents=True)
    np.save(directory / "run.npy", np.zeros((CHANNELS, 4 * RATE), np.float16))
    heard = [("The", 0.5), ("cat", 1.0), ("the", 1.5), ("zebra", 3.9)]
    rows = [("run", onset, word, 0, 3.9) for word, onset in heard]
    pd.DataFrame(rows, columns=["run", "onset", "word", "sentence", "end"]).to_csv(directory / WORDS, sep="\t")


def test_prototypes_encode() -> None:
    encoded = Prototypes.encode(("ab", "abcd"), Tokenizer(), Encoder(), 1)
    assert encoded.vectors[:, 0].tolist() == [2 * 2.5, 2 * 4.5]


def test_lexicon_encode() -> None:
    words = pd.DataFrame({"word": ["Abc", "ab", "abc"], "token": [7, -2, 7]})
    lexicon = Lexicon.encode(words, Tokenizer(), Encoder(), 1)
    assert lexicon.keys.tolist() == [-2, 7]
    assert lexicon.vectors[:, 0].tolist() == [2 * 2.5, 2 * 3.5]


def test_prototypes_path(tmp_path: Path) -> None:
    assert Prototypes.path(Root(tmp_path)).parent == tmp_path / "prototypes"  # type: ignore[arg-type]


def test_lexicon_path(tmp_path: Path) -> None:
    assert Lexicon.path(Root(tmp_path)).parent == tmp_path / "prototypes"  # type: ignore[arg-type]
    assert Lexicon.path(Root(tmp_path)) != Prototypes.path(Root(tmp_path))  # type: ignore[arg-type]


def test_prototypes_save(tmp_path: Path) -> None:
    assert Prototypes(torch.ones(2, DIM)).save(tmp_path / "p.npy").exists()


def test_lexicon_save(tmp_path: Path) -> None:
    assert lexicon().save(tmp_path / "nested" / "l.npz").exists()


def test_prototypes_load(tmp_path: Path) -> None:
    path = Prototypes(torch.ones(2, DIM)).save(tmp_path / "p.npy")
    assert torch.equal(Prototypes.load(path).vectors, torch.ones(2, DIM))


def test_lexicon_load(tmp_path: Path) -> None:
    loaded = Lexicon.load(lexicon().save(tmp_path / "l.npz"))
    assert torch.equal(loaded.keys, lexicon().keys)
    assert torch.equal(loaded.vectors, lexicon().vectors)


def test_to() -> None:
    assert lexicon().to(CPU).vectors.device == CPU


def test_rows() -> None:
    assert lexicon().rows(torch.tensor([[9, -5], [4, 100]])).tolist() == [[2, 0], [-1, -1]]
    assert lexicon().rows(torch.tensor([-100])).tolist() == [-1]


def test_prototypes_write(tmp_path: Path) -> None:
    written = Prototypes.write(Root(tmp_path), Tokenizer(), Encoder())  # type: ignore[arg-type]
    assert Prototypes.load(written).vectors.shape == (len(PRIMARY), DIM)


def test_lexicon_write(tmp_path: Path) -> None:
    cache(tmp_path)
    root = Root(tmp_path)
    written = Lexicon.load(Lexicon.write(root, Tokenizer(), Encoder()))  # type: ignore[arg-type]
    the, cat = pd.util.hash_array(np.array(["the", "cat"], dtype=object)).view(np.int64).tolist()
    assert written.keys.tolist() == sorted([the, cat])
    assert written.vectors.shape == (2, DIM)


def test_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, "AutoTokenizer", Loader(Tokenizer()))
    monkeypatch.setattr(module, "T5EncoderModel", Loader(Encoder()))
    monkeypatch.setattr(module.DataRoot, "from_config", classmethod(lambda _: Root(tmp_path)))
    monkeypatch.setattr("sys.argv", ["prototypes"])
    Prototypes.main()
    assert Prototypes.load(Prototypes.path(Root(tmp_path))).vectors.shape == (len(PRIMARY), DIM)  # type: ignore[arg-type]
    cache(tmp_path)
    monkeypatch.setattr("sys.argv", ["prototypes", "--lexicon"])
    Prototypes.main()
    assert Lexicon.load(Lexicon.path(Root(tmp_path))).keys.shape == (2,)  # type: ignore[arg-type]
