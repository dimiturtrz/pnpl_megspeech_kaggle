"""Every source's recordings converted into LibriBrain's layout: a run's `.h5` and its `events.tsv`."""

from enum import StrEnum


class Event(StrEnum):
    """A column of a run's `events.tsv`: pnpl's schema, which every raw source here is written into."""

    KIND = "kind"
    SEGMENT = "segment"
    SENTENCE = "sentenceidx"
    WORD = "wordidx"
    PHONEME = "phonemeidx"
    ONSET = "timemeg"
    DURATION = "duration"
    CHAPTER = "timechapter"


class Kind(StrEnum):
    """What an `events.tsv` row marks: a heard word or one of its phonemes."""

    WORD = "word"
    PHONEME = "phoneme"
