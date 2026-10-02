"""Pull the Deep-track data into `raw/pnpl_libribrain`, laid out exactly as `pnpl`'s own downloader would.

`pnpl` resolves every file as `<data_path>/<repo-relative path>`, so mirroring the HuggingFace tree under one
root lets its loaders find the files instead of re-fetching them. Only subject 0's preprocessed `.h5` and
events `.tsv` are pulled; the raw `.fif` (hundreds of GB) is never needed.

    uv run python -m megspeech.fetch            # holdout, then LibriBrain, then LibriBrain2
    uv run python -m megspeech.fetch --others   # subjects 1-32 (~44 min each), for cross-subject training
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import snapshot_download

from core.paths import DataRoot

LOG = logging.getLogger(__name__)

SUBJECT0 = ("*sub-0_*_meg.h5", "*sub-0_*_events.tsv")
"""Subject 0's preprocessed recordings and their word/phoneme events, in any task directory."""

OTHER_SUBJECTS = ("*sub-[1-9]*_meg.h5", "*sub-[1-9]*_events.tsv")
"""Subjects 1-32's two Sherlock runs each; never subject 0, whose files `SUBJECT0` already covers."""

PARALLEL_FILES = 8
"""Concurrent file downloads; HF serves each file from a CDN, so throughput scales with streams."""


@dataclass(frozen=True)
class Source:
    """One HuggingFace dataset repo and the files of it the Deep track needs."""

    repo: str
    patterns: tuple[str, ...]

    def fetch(self, destination: Path) -> Path:
        """Mirror this repo's matching files into `destination`; files already present are skipped."""
        LOG.info("fetching %s %s -> %s", self.repo, self.patterns, destination)
        return Path(
            snapshot_download(
                repo_id=self.repo,
                repo_type="dataset",
                allow_patterns=list(self.patterns),
                local_dir=destination,
                max_workers=PARALLEL_FILES,
                token=False,
            )
        )


HOLDOUT = Source("pnpl/LibriBrain-Competition-2026", ("*.npz",))
SOURCES = (
    HOLDOUT,
    Source("pnpl/LibriBrain", SUBJECT0),
    Source("pnpl/LibriBrain2", SUBJECT0),
)
"""Fetch order: the 5.7 GB holdout first, so submission plumbing can start before the training data lands."""

OTHERS = Source("pnpl/LibriBrain2", OTHER_SUBJECTS)


def main() -> None:
    """Fetch every source (or only the holdout, or only the other subjects) into the competition's raw directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--holdout-only", dest="sources", action="store_const", const=(HOLDOUT,), default=SOURCES)
    choice.add_argument("--others", dest="sources", action="store_const", const=(OTHERS,))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    destination = DataRoot.from_config().raw()
    for source in args.sources:
        source.fetch(destination)


if __name__ == "__main__":
    main()
