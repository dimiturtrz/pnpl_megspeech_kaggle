import numpy as np

from megspeech.data.spans.spans import MAX_WORDS, PAD
from megspeech.evaluation.studies.wikispans import OOV, Tokens, Wikitext

LINES = [
    " = Alpha = \n",
    " It 's the cat . \n",
    " = = Part = = \n",
    " Is it ? \n",
    " = Sherlock Holmes = \n",
    " The end . \n",
]


def test_articles() -> None:
    articles = Wikitext(LINES).articles()
    assert [len(article) for article in articles] == [4, 2]


def test_kept() -> None:
    kept, dropped = Wikitext(LINES).kept()
    assert dropped == 1
    assert kept[0][0] == LINES[0]


def tokens() -> Tokens:
    return Tokens({"it’s": 0, "the": 1, "is": 2, "it": 3})


def test_words() -> None:
    assert Tokens.words(" It 's the cat . ") == ["It's", "the", "cat", "."]
    assert Tokens.words(" do n't ' go") == ["don't", "'", "go"]
    assert Tokens.words(" = = Part = = ") == []


def test_sentences() -> None:
    assert list(tokens().sentences(LINES[:4])) == [[0, 1, OOV], [2, 3]]
    assert list(tokens().sentences([" 1990 , the "])) == [[1]]


def test_flat() -> None:
    states, sizes = tokens().flat([LINES[:4]])
    assert states.tolist() == [0, 1, OOV, 2, 3]
    assert sizes.tolist() == [3, 2]


def test_spans() -> None:
    states = np.arange(MAX_WORDS + 3, dtype=np.int8)
    spans = Tokens.spans(states, np.array([MAX_WORDS + 1, 2]))
    assert spans.shape[1] == MAX_WORDS
    assert int((spans != PAD).sum()) == MAX_WORDS + 3
    assert spans[-1, :2].tolist() == [MAX_WORDS + 1, MAX_WORDS + 2]
