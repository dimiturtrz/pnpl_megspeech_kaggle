from megspeech.data.cache.kaggle import Track
from megspeech.submission.config import ReportConfig


def test_report_config_asks_for_nothing_by_default() -> None:
    report = ReportConfig()
    assert not report.local
    assert report.tracks == ()
    assert not report.aligned


def test_report_config_records_what_it_is_given() -> None:
    track = next(iter(Track))
    report = ReportConfig(local=True, tracks=(track,), aligned=True)
    assert report.local
    assert report.tracks == (track,)
    assert report.aligned
