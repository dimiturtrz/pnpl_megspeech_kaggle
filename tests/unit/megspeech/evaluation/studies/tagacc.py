import numpy as np

from megspeech.data.labels.phonemes import UNKNOWN
from megspeech.evaluation.studies.tagacc import Headed


def test_scored() -> None:
    headed = Headed(
        np.array([1, UNKNOWN, 2, 1]),
        np.array([1, 5, 0, 1]),
        np.array([True, True, False, False]),
        np.array(["x", "x", "y", "y"]),
    )
    scored, predicted = headed.scored()
    assert predicted.tolist() == [1, 0, 1]
    assert scored.scores(predicted)["all"][1:] == (2 / 3, 3)
