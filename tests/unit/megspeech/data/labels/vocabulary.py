import numpy as np
import pandas as pd

from megspeech.data.labels.vocabulary import ABSENT, PRIMARY, VOCAB_SIZE, Vocabulary

WORDS = Vocabulary(("the", "it’s", "a"))


def test_spelled() -> None:
    assert Vocabulary.spelled(pd.Series(["It's,", "THE", "a-b"])).tolist() == ["it’s", "the", "ab"]


def test_spell() -> None:
    assert Vocabulary.spell("It's,") == "it’s"


def test_labels() -> None:
    assert WORDS.labels(pd.Series(["The", "it's", "zebra"])).tolist() == [0, 1, ABSENT]
    assert WORDS.labels([]).dtype == np.int64
    assert WORDS.labels([]).tolist() == []


def test_token() -> None:
    token = Vocabulary.token(pd.Series(["The", "the", "a"]))
    assert token.dtype == np.int64
    assert token[0] == token[1] != token[2]


def test_len() -> None:
    assert len(WORDS) == len(WORDS.positions) == 3
    assert len(PRIMARY) == VOCAB_SIZE
