from typing import cast

import pytest
from torch import nn

from megspeech.evaluation.studies.ablation import ALL_ON, Sources
from megspeech.model.branch import Branch
from megspeech.model.fusion import Fusion


class Stub(nn.Module):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name


def fused(*names: str) -> Fusion:
    return Fusion([cast("Branch", Stub(name)) for name in names], 0.0)


def test_sources_of() -> None:
    assert Sources.of(nn.Sequential(fused("a"), fused("a"))).names() == ("primary", "a")
    with pytest.raises(ValueError, match="no fusion branches"):
        Sources.of(nn.Sequential(fused()))
    with pytest.raises(ValueError, match="different sources"):
        Sources.of(nn.Sequential(fused("a"), fused("b")))


def test_names() -> None:
    assert Sources.of(fused("a", "b")).names() == ("primary", "a", "b")


def test_silenced() -> None:
    """Every member silences the source inside the block and none after it."""
    sources = Sources.of(nn.Sequential(fused("a"), fused("a")))
    with sources.silenced("a"):
        assert [f.silent for f in sources.fusions] == ["a", "a"]
    assert [f.silent for f in sources.fusions] == ["", ""]
    with pytest.raises(KeyError), sources.silenced("z"):
        pass


def test_sweep() -> None:
    """The measure runs all on, then once per source with that source silent."""
    sources = Sources.of(fused("a"))
    seen = sources.sweep(lambda: sources.fusions[0].silent)
    assert seen == {ALL_ON: "", "primary": "primary", "a": "a"}
