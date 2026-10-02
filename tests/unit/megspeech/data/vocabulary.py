import pandas as pd

from megspeech.data.vocabulary import ABSENT, PRIMARY, Vocabulary

WORDS = Vocabulary(("the", "it’s", "a"))


def test_spell() -> None:
    assert Vocabulary.spell("It's,") == "it’s"


def test_labels() -> None:
    assert WORDS.labels(pd.Series(["The", "it's", "zebra"])).tolist() == [0, 1, ABSENT]


def test_len() -> None:
    assert len(WORDS) == len(WORDS.positions) == 3
    assert len(PRIMARY) == 50
