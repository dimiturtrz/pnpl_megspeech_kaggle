import numpy as np
import pandas as pd
import torch

from megspeech.data.labels.vocabulary import IGNORE, PRIMARY
from megspeech.evaluation.studies.lmtext import Sentences


def sentences(words: list[str]) -> Sentences:
    return Sentences(pd.DataFrame({"run": "r", "word": words, "left": [()] * len(words), "right": [()] * len(words)}))


def test_labels() -> None:
    first = PRIMARY.words[3]
    assert Sentences.labels(pd.Series([first.upper(), "zzzzqq"])).tolist() == [3, IGNORE]


def test_len() -> None:
    assert len(sentences(["a", "b"])) == 2


def test_agrees() -> None:
    first = PRIMARY.words[3]
    assert sentences([first, "zzzzqq"]).agrees(np.array([3, IGNORE])) == 1.0
    assert sentences([first, "zzzzqq"]).agrees(np.array([3, 4])) == 0.5


def test_joined() -> None:
    joined = Sentences.joined([sentences(["a"]), sentences(["b", "c"])])
    assert joined.frame["word"].tolist() == ["a", "b", "c"]
    assert joined.frame.index.tolist() == [0, 1, 2]


def test_contexts() -> None:
    frame = pd.DataFrame({"run": "r", "word": ["a", "b"], "left": [(), ("a",)], "right": [("b",), ()]})
    out = Sentences(frame).contexts(torch.tensor([1, 0]))
    assert out == [(("a",), ()), ((), ("b",))]
