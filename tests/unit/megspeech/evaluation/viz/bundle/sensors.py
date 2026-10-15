import mne

from megspeech.data.raw.sources.masc import Neuromag
from megspeech.evaluation.viz.bundle.sensors import REGIONS, Helmet

NAMES = ("MEG 0111", "MEG 0112", "MEG 0113", "MEG 2421")


def test_layout() -> None:
    info = mne.create_info(list(NAMES), 1000.0, ["mag", "grad", "grad", "mag"])
    layout = Helmet.layout(Neuromag(info))
    assert layout["names"] == [name.replace(" ", "") for name in NAMES]
    assert layout["kinds"] == ["mag", "grad", "grad", "mag"]
    assert layout["xy"] == Helmet.plane([name.replace(" ", "") for name in NAMES])
    regions = layout["regions"]
    assert isinstance(regions, dict)
    assert set(regions) == set(REGIONS)
    assert regions["Left-temporal"] == [0, 1, 2]
    assert regions["Right-temporal"] == [3]


def test_plane() -> None:
    left, right, off = Helmet.plane(["MEG0111", "MEG2421", "MEG9999"])
    assert left is not None
    assert right is not None
    assert left[0] < right[0]
    assert all(0 <= v <= 1 for v in (*left, *right))
    assert off is None
