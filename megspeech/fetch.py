"""Pull the Deep-track data into `raw/pnpl_libribrain`, laid out exactly as `pnpl`'s own downloader would.

`pnpl` resolves every file as `<data_path>/<repo-relative path>`, so mirroring the HuggingFace tree under one
root lets its loaders find the files instead of re-fetching them. Only subject 0's preprocessed `.h5` and
events `.tsv` are pulled; the raw `.fif` (hundreds of GB) is never needed.

    uv run python -m megspeech.fetch            # holdout, then LibriBrain, then LibriBrain2
    uv run python -m megspeech.fetch --others   # subjects 1-32 (~44 min each), for cross-subject training
    uv run python -m megspeech.fetch --weights  # pretrained MEG-XL and a GPT-2-class LM, into models/foreign
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import override

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
    """One HuggingFace dataset repo and the files of it the Deep track needs, mirrored into the raw directory."""

    repo: str
    patterns: tuple[str, ...]
    kind: str = "dataset"

    def home(self, root: DataRoot) -> Path:
        return root.raw()

    def fetch(self, root: DataRoot) -> Path:
        """Mirror this repo's matching files into its home; files already present are skipped."""
        destination = self.home(root)
        LOG.info("fetching %s %s -> %s", self.repo, self.patterns, destination)
        return Path(
            snapshot_download(
                repo_id=self.repo,
                repo_type=self.kind,
                allow_patterns=list(self.patterns),
                local_dir=destination,
                max_workers=PARALLEL_FILES,
                token=False,
            )
        )


@dataclass(frozen=True)
class Weights(Source):
    """A third-party model repo, kept in its own folder under the foreign models."""

    kind: str = "model"

    @override
    def home(self, root: DataRoot) -> Path:
        return root.foreign(self.repo.split("/")[-1])


HOLDOUT = Source("pnpl/LibriBrain-Competition-2026", ("*.npz",))
SOURCES = (
    HOLDOUT,
    Source("pnpl/LibriBrain", SUBJECT0),
    Source("pnpl/LibriBrain2", SUBJECT0),
)
"""Fetch order: the 5.7 GB holdout first, so submission plumbing can start before the training data lands."""

OTHERS = Source("pnpl/LibriBrain2", OTHER_SUBJECTS)

WEIGHTS = (
    Weights("pnpl/MEG-XL", ("*",)),
    Weights("openai-community/gpt2-large", ("*.json", "*.safetensors", "*.txt")),
)
"""MEG-XL as a pretrained init or ensemble seat (7gf.5); a GPT-2-class LM for the prior fusion (7gf.6)."""


def main() -> None:
    """Fetch every source, or only the holdout, the other subjects, or the pretrained weights."""
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--holdout-only", dest="sources", action="store_const", const=(HOLDOUT,), default=SOURCES)
    choice.add_argument("--others", dest="sources", action="store_const", const=(OTHERS,))
    choice.add_argument("--weights", dest="sources", action="store_const", const=WEIGHTS)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    root = DataRoot.from_config()
    for source in args.sources:
        source.fetch(root)


if __name__ == "__main__":
    main()
