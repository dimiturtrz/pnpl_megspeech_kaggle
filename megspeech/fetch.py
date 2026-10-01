"""Pull the Deep-track data into `raw/pnpl_libribrain`, laid out exactly as `pnpl`'s own downloader would.

`pnpl` resolves every file as `<data_path>/<repo-relative path>`, so mirroring the HuggingFace tree under one
root lets its loaders find the files instead of re-fetching them. Only subject 0's preprocessed `.h5` and
events `.tsv` are pulled; the raw `.fif` (hundreds of GB) is never needed.

    uv run python -m megspeech.fetch            # holdout, then LibriBrain, then LibriBrain2
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


def main() -> None:
    """Fetch every source (or only the holdout) into the competition's raw directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout-only", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    destination = DataRoot.from_config().raw()
    for source in (HOLDOUT,) if args.holdout_only else SOURCES:
        source.fetch(destination)


if __name__ == "__main__":
    main()
