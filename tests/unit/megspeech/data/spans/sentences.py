import pandas as pd

from megspeech.data.spans.grouping import KAGGLE_WORDS
from megspeech.data.spans.sentences import Sentences

WORDS = pd.DataFrame({"run": ["a", "a", "a", "b"], "sentence": [0, 0, 1, 0], "onset": [0.0, 0.2, 0.4, 0.1]})


def test_sentences() -> None:
    """A sentence cut keys sentences by run too: the same sentence number in two runs is two spans."""
    assert Sentences()(WORDS).lengths.tolist() == [2, 1, 1]
    assert Sentences(KAGGLE_WORDS).words == KAGGLE_WORDS
