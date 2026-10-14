"""Run one study by its `<group>.<study>` name; the rest of the command line is that study's own.

uv run python -m megspeech.evaluation.studies                       # lists every study
uv run python -m megspeech.evaluation.studies errors.gap m11xsp7hub6d2v3 --seed 1
"""

import pkgutil
import runpy
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import megspeech.evaluation.studies as package

GUARD = 'if __name__ == "__main__":'


@dataclass(frozen=True)
class Studies:
    """The runnable modules under one package: those that guard a `__main__` entry."""

    root: str = package.__name__
    path: Sequence[str] = tuple(package.__path__)

    def names(self) -> list[str]:
        """Every study as `<group>.<study>`, sorted."""
        found = [info for info in pkgutil.walk_packages(self.path, f"{self.root}.") if not info.ispkg]
        runnable = [
            info.name
            for info in found
            if not info.name.endswith("__main__") and GUARD in self.source(info).read_text(encoding="utf-8")
        ]
        return sorted(name.removeprefix(f"{self.root}.") for name in runnable)

    @staticmethod
    def source(info: pkgutil.ModuleInfo) -> Path:
        return Path(getattr(info.module_finder, "path", "")) / f"{info.name.rpartition('.')[2]}.py"

    def run(self, argv: Sequence[str]) -> None:
        """Runs the study `argv[0]` names as `__main__`, the rest its arguments; lists them all when there is none."""
        if not argv:
            sys.stdout.write("\n".join(self.names()) + "\n")
            return
        sys.argv = [f"{self.root} {argv[0]}", *argv[1:]]
        runpy.run_module(f"{self.root}.{argv[0]}", run_name="__main__", alter_sys=True)


def main() -> None:
    Studies().run(sys.argv[1:])


if __name__ == "__main__":
    main()
