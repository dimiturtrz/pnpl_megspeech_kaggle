from megspeech.data.epochs import DECIMATE, RATE, SFREQ, WINDOW


def test_rates() -> None:
    """The decimation is an integer factor, and a 1 s window is `RATE` cached samples."""
    assert RATE * DECIMATE == SFREQ
    assert WINDOW // DECIMATE == RATE
