import pkgutil
import sys
from pathlib import Path

import pytest

from megspeech.evaluation.studies.__main__ import Studies

STUDY = """import sys
from pathlib import Path

if __name__ == "__main__":
    Path(sys.argv[1]).write_text(" ".join(sys.argv[2:]))
"""
REWRITER = "AssertionRewritingHook"
"""pytest's import hook claims every `*.py` here and cannot hand runpy the code of a module it rewrote."""


@pytest.fixture
def tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Studies:
    group = tmp_path / "toy" / "group"
    group.mkdir(parents=True)
    (tmp_path / "toy" / "__init__.py").write_text("")
    (group / "__init__.py").write_text("")
    (group / "study.py").write_text(STUDY)
    (group / "helper.py").write_text("VALUE = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(sys, "argv", list(sys.argv))
    monkeypatch.setattr(sys, "meta_path", [hook for hook in sys.meta_path if type(hook).__name__ != REWRITER])
    return Studies(root="toy", path=(str(tmp_path / "toy"),))


def test_names(tree: Studies) -> None:
    assert tree.names() == ["group.study"]


def test_names_finds_the_real_studies() -> None:
    names = Studies().names()
    assert {"errors.gap", "errors.cause"} <= set(names)
    assert all(name.count(".") == 1 for name in names)


def test_source(tree: Studies) -> None:
    info = next(info for info in pkgutil.walk_packages(tree.path, "toy.") if info.name == "toy.group")
    assert Studies.source(info).name == "group.py"


def test_run_lists_without_a_study(tree: Studies, capsys: pytest.CaptureFixture[str]) -> None:
    tree.run([])
    assert capsys.readouterr().out.split() == ["group.study"]


def test_run_hands_the_rest_to_the_study(tree: Studies, tmp_path: Path) -> None:
    seen = tmp_path / "seen.txt"
    tree.run(["group.study", str(seen), "--seed", "1"])
    assert seen.read_text() == "--seed 1"
