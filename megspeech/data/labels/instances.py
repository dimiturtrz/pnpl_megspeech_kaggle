"""Which token instance of a MASC story each training word is: every MASC subject heard the same four stories, and every
run of a story holds the same words in the same order, so a story and a word's place in its run's word table name one
spoken token whoever heard it. Every other corpus's words are no instance."""

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from jaxtyping import Int

from megspeech.data.raw.runs import NAME, Column

STORY = re.compile(r"MASC(?P<story>\d+)")
"""The task `megspeech.data.raw.masc.Recording.name` gives a MASC story's run."""
NO_INSTANCE = -1


@dataclass(frozen=True)
class Instances:
    """Each word's token `code`, the same for every subject's hearing of it and `NO_INSTANCE` outside MASC, and the
    `subject` who heard it (`NO_INSTANCE` for a run name that names none)."""

    code: Int[np.ndarray, " n"]
    subject: Int[np.ndarray, " n"]

    @classmethod
    def of(cls, words: pd.DataFrame) -> "Instances":
        """The instances of `words` (`Corpus.positions`): its `run` names and each word's `position` in its run."""
        names = pd.Series(words[Column.RUN].unique(), dtype=object)
        parsed = pd.Series([f"{name}_" for name in names]).str.extract(NAME)
        stories = pd.to_numeric(parsed["task"].str.extract(f"^{STORY.pattern}$")["story"])
        position = words[Column.POSITION].to_numpy(np.int64)
        stride = int(position.max()) + 1 if len(position) else 0
        story_of = words[Column.RUN].map(pd.Series(stories.to_numpy(), index=names)).to_numpy(np.float64)
        subject_of = words[Column.RUN].map(pd.Series(pd.to_numeric(parsed["subject"]).to_numpy(), index=names))
        code = np.where(np.isnan(story_of), NO_INSTANCE, np.nan_to_num(story_of) * stride + position)
        subject = subject_of.fillna(NO_INSTANCE).to_numpy(np.int64, copy=True)
        return cls(code.astype(np.int64), subject)

    @classmethod
    def sentences(cls, words: pd.DataFrame) -> Int[np.ndarray, " n"]:
        """Each word's sentence code among `words` (`Corpus.positions`), the same for every subject's hearing of a
        MASC sentence (its story and its first word's position) and outside MASC its run's own `sentence`."""
        code = cls.of(words).code
        position = words[Column.POSITION].to_numpy(np.int64)
        opening = (
            words.groupby([Column.RUN, Column.SENTENCE], sort=False)[Column.POSITION]
            .transform("min")
            .to_numpy(np.int64)
        )
        masc = code != NO_INSTANCE
        keys = pd.DataFrame(
            {
                Column.RUN: np.where(masc, "", words[Column.RUN].to_numpy(object)),
                "at": np.where(masc, code - position + opening, words[Column.SENTENCE].to_numpy(np.int64)),
            }
        )
        return keys.groupby([Column.RUN, "at"], sort=False).ngroup().to_numpy(np.int64, copy=True)

    @property
    def heard(self) -> int:
        """Words that are a MASC instance."""
        return int((self.code != NO_INSTANCE).sum())
