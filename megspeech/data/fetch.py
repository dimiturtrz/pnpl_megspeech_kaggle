"""Pull the Deep-track data into `raw/pnpl_libribrain`, laid out exactly as `pnpl`'s own downloader would.

`pnpl` resolves every file as `<data_path>/<repo-relative path>`, so mirroring the HuggingFace tree under one
root lets its loaders find the files instead of re-fetching them. Only subject 0's preprocessed `.h5` and
events `.tsv` are pulled; the raw `.fif` (hundreds of GB) is never needed.

    uv run python -m megspeech.data.fetch            # holdout, then LibriBrain, then LibriBrain2
    uv run python -m megspeech.data.fetch --others   # subjects 1-32 (~44 min each), for cross-subject training
    uv run python -m megspeech.data.fetch --weights  # pretrained MEG-XL and a GPT-2-class LM, into models/foreign,
                                                     # and MEG-XL's code at its pin, into external/
"""

import argparse
import logging
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import override
from urllib.request import urlopen
from zipfile import ZipFile

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


EXTERNAL = Path(__file__).resolve().parents[2] / "external"
"""The gitignored checkouts of third-party code at the repo root, never vendored into git."""

PINNED = ".commit"
"""The file a checkout records its commit in; a checkout already at its pin is not fetched again."""


@dataclass(frozen=True)
class Checkout:
    """A third-party GitHub repo's code at one pinned commit, extracted from its archive into `external/<name>`."""

    repo: str
    commit: str
    parent: Path = EXTERNAL

    def home(self) -> Path:
        return self.parent / self.repo.split("/")[-1]

    def archive(self) -> str:
        return f"https://github.com/{self.repo}/archive/{self.commit}.zip"

    def fetch(self, _root: DataRoot) -> Path:
        """Extract the pinned commit's tree into its home, its top-level folder stripped."""
        home = self.home()
        marker = home / PINNED
        if marker.exists() and marker.read_text().strip() == self.commit:
            return home
        LOG.info("fetching %s@%s -> %s", self.repo, self.commit, home)
        with urlopen(self.archive()) as response, ZipFile(BytesIO(response.read())) as archive:
            for member in archive.infolist():
                relative = Path(*Path(member.filename).parts[1:])
                if member.is_dir() or not relative.parts:
                    continue
                target = home / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(member))
        marker.write_text(self.commit)
        return home


@dataclass(frozen=True)
class Download:
    """One file at a pinned URL, saved as `raw/<dataset>/<name>`."""

    url: str
    dataset: str

    def home(self, root: DataRoot) -> Path:
        return root.raw(self.dataset) / self.url.rsplit("/", 1)[-1]

    def fetch(self, root: DataRoot) -> Path:
        """Save the file unless it is already there."""
        home = self.home(root)
        if not home.exists():
            LOG.info("fetching %s -> %s", self.url, home)
            home.parent.mkdir(parents=True, exist_ok=True)
            with urlopen(self.url) as response:
                home.write_bytes(response.read())
        return home


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

MEGXL = Checkout("neural-processing-lab/MEG-XL", "97da04171d22e8fe4822cfeb3047a57ecf86b318")
"""The code MEG-XL's weights load into (MIT): its criss-cross trunk and BioCodec tokenizer."""

NEUROMAG = Download(
    "https://raw.githubusercontent.com/mne-tools/mne-python/v1.10.0/mne/io/tests/data/test_raw.fif", "mne"
)
"""A Neuromag-306 recording from MNE's tests (BSD): its channels' device-frame positions and coil orientations, which
are the system's and the same in every recording, are the sensor geometry MEG-XL embeds and LibriBrain's files omit."""


def main() -> None:
    """Fetch every source, or only the holdout, the other subjects, or the pretrained weights."""
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--holdout-only", dest="sources", action="store_const", const=(HOLDOUT,), default=SOURCES)
    choice.add_argument("--others", dest="sources", action="store_const", const=(OTHERS,))
    choice.add_argument("--weights", dest="sources", action="store_const", const=(*WEIGHTS, MEGXL, NEUROMAG))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    root = DataRoot.from_config()
    for source in args.sources:
        source.fetch(root)


if __name__ == "__main__":
    main()
