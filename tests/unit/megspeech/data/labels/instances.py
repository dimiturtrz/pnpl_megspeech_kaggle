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


def test_sentences() -> None:
    """Every subject's hearing of a MASC sentence shares its code, whatever its run's own sentence index; another
    story's sentence or another corpus's run does not."""
    words = pd.DataFrame(
        {
            "run": ["sub-101_ses-0_task-MASC1_run-1"] * 3
            + ["sub-102_ses-0_task-MASC1_run-1"] * 3
            + ["sub-101_ses-0_task-MASC2_run-1", "sub-0_ses-1_task-Sherlock1_run-1", "a"],
            "position": [0, 1, 2, 0, 1, 2, 0, 0, 0],
            "sentence": [0, 0, 1, 5, 5, 6, 0, 0, 0],
        }
    )
    code = Instances.sentences(words).tolist()
    assert code[0] == code[1] == code[3] == code[4]
    assert code[2] == code[5] != code[0]
    assert len({code[0], code[2], code[6], code[7], code[8]}) == 5


def test_heard() -> None:
    assert Instances.of(WORDS).heard == 4
    assert Instances.of(WORDS.iloc[4:]).heard == 0
