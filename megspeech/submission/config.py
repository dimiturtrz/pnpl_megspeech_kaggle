"""What a training run reports in its own process once it is saved."""

from dataclasses import dataclass

from megspeech.data.cache.kaggle import Track


@dataclass(frozen=True)
class ReportConfig:
    """What a training run reports in its own process once it is saved; the default asks for nothing."""

    local: bool = False
    """Score val and the local test at val's tau, as `megspeech.submission --local` does."""
    tracks: tuple[Track, ...] = ()
    """Tracks whose submission is written under models/<name>-<track>, as `megspeech.submission --track` does."""
    aligned: bool = False
    """Read the Broad track's other subjects re-coloured to subject 0's covariance (`Scoring.aligned`)."""
