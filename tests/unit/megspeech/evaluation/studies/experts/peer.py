from pathlib import Path

from core.paths import MODELS, DataRoot
from megspeech.evaluation.studies.experts.peer import OUT, Dump


def test_path(tmp_path: Path) -> None:
    path = Dump.path(DataRoot(tmp_path), "base")
    assert path.name == "base.pt"
    assert path.parent == DataRoot(tmp_path).models() / OUT
    assert MODELS in path.parts
