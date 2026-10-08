"""Load every saved run config under the models root, to prove a config-schema change keeps the archive readable.

uv run python -m megspeech.training.cli.loadall > loadall.tsv      # one `name<TAB>reason` line per failing run
"""

import logging
import sys
from collections.abc import Iterable
from pathlib import Path

from core.paths import DataRoot
from megspeech.training.config import METRICS, RunConfig

LOG = logging.getLogger(__name__)
NEWLINE = chr(10)


class LoadAll:
    """The saved runs under a directory, each loaded through `RunConfig.saved`."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def runs(self) -> list[Path]:
        """The run directories holding a metrics record, by name."""
        return sorted(path.parent for path in self.root.glob(f"*/{METRICS}"))

    @staticmethod
    def failure(directory: Path) -> str | None:
        """Why the run's config does not load, or None when it does."""
        try:
            RunConfig.saved(directory)
        except (KeyError, ValueError, TypeError) as error:
            return f"{type(error).__name__}: {error}"
        return None

    def failures(self) -> dict[str, str]:
        """Each failing run's name and reason."""
        reasons = {run.name: self.failure(run) for run in self.runs()}
        return {name: reason for name, reason in reasons.items() if reason is not None}

    @staticmethod
    def report(loaded: int, failing: Iterable[tuple[str, str]]) -> list[str]:
        """The `name<TAB>reason` lines, led by the loaded count."""
        return [f"loaded\t{loaded}", *(f"{name}\t{reason}" for name, reason in failing)]


def main() -> None:
    """Print the loaded count and each failing run with its reason."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    archive = LoadAll(DataRoot.from_config().models())
    failing = archive.failures()
    loaded = len(archive.runs()) - len(failing)
    LOG.info("%d loaded, %d failing", loaded, len(failing))
    lines = LoadAll.report(loaded, failing.items())
    sys.stdout.write(NEWLINE.join(lines) + NEWLINE)


if __name__ == "__main__":
    main()
