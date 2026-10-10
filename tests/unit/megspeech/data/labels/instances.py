import pandas as pd

from megspeech.data.labels.instances import NO_INSTANCE, Instances

WORDS = pd.DataFrame(
    {
        "run": [
            "sub-101_ses-0_task-MASC1_run-1",
            "sub-101_ses-0_task-MASC1_run-1",
            "sub-102_ses-1_task-MASC1_run-1",
            "sub-102_ses-0_task-MASC2_run-1",
            "sub-0_ses-1_task-Sherlock1_run-1",
            "a",
        ],
        "position": [0, 2, 2, 2, 2, 1],
    }
)


def test_of() -> None:
    """A story's position names one instance across subjects and sessions; another story's or corpus's does not."""
    instances = Instances.of(WORDS)
    code = instances.code.tolist()
    assert code[1] == code[2] != code[3]
    assert code[0] != code[1]
    assert code[4:] == [NO_INSTANCE, NO_INSTANCE]
    assert instances.subject.tolist() == [101, 101, 102, 102, 0, NO_INSTANCE]


def test_heard() -> None:
    assert Instances.of(WORDS).heard == 4
    assert Instances.of(WORDS.iloc[4:]).heard == 0
