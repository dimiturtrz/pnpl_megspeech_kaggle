import numpy as np
import pandas as pd

from megspeech.evaluation.text import Text


def table(run: str, words: tuple[str, ...] = ("The", "man", "said", "hello")) -> pd.DataFrame:
    return pd.DataFrame(
        {"run": run, "onset": np.arange(len(words)) * 0.3, "word": list(words), "sentence": [0, 0, 1, 1]}
    )


def test_of() -> None:
    assert Text.of(pd.concat([table("a"), table("b")])).words["position"].tolist() == [0, 1, 0, 1] * 2


def test_keys() -> None:
    assert Text.of(table("a")).keys[1] == (0, 1, "man")
    assert Text.of(table("a")).keys.is_unique


def test_matches() -> None:
    reference = Text.of(table("a"))
    other = Text(Text.of(table("b", words=("The", "boy", "said", "hello"))).words.iloc[::-1])
    assert reference.matches(other).tolist() == [3, -1, 1, 0]
