"""Downloads the LibriSpeech alignment shards and metadata archives into the raw directory.

Sources and licences are listed in `megspeech.data.labels.librispeech`.
"""

import logging
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

from core.paths import DataRoot

LOG = logging.getLogger(__name__)

ALIGNMENTS_REPO = "datasets/gilkeyio/librispeech-alignments/data"
"""Where the aligned parquet shards live on the hub."""

DATASET = "librispeech_alignments"
"""The dataset's slug under `raw/` (shards + metadata) and `processed/` (the span table built from them)."""
DOWNLOAD_THREADS = 4

METADATA_ARCHIVES = {
    "https://www.openslr.org/resources/12/raw-metadata.tar.gz": (
        "CHAPTERS.TXT",
        "lv-annotated.sqlite.db",
        "LICENSE.TXT",
    ),
    "https://www.openslr.org/resources/145/manifests.tar.gz": (
        "train-clean-100.json",
        "train-clean-360.json",
        "train-other-500.json",
        "dev-clean.json",
        "dev-other.json",
        "test-clean.json",
        "test-other.json",
    ),
}
"""Archive URL to the members kept from it, flattened into the metadata directory."""


class Fetch:
    """Downloads the alignment shards and the metadata archives into the raw directory."""

    def __init__(self, root: DataRoot) -> None:
        self.home = root.raw(DATASET)
        self.meta = self.home / "meta"

    def alignments(self, source: HfFileSystem) -> list[Path]:
        """Each remote shard's `id`, `transcript` and `words` columns as a local parquet; present ones are skipped."""
        self.home.mkdir(parents=True, exist_ok=True)
        remote = sorted(source.glob(f"{ALIGNMENTS_REPO}/*.parquet"))
        with ThreadPoolExecutor(DOWNLOAD_THREADS) as pool:
            return list(pool.map(lambda path: self.shard(source, path), remote))

    def shard(self, source: HfFileSystem, remote: str) -> Path:
        target = self.home / Path(remote).name
        if target.exists():
            return target
        with source.open(remote, "rb") as handle:
            table = pq.ParquetFile(handle).read(columns=["id", "transcript", "words"])
        pq.write_table(table, target.with_suffix(".part"), compression="zstd")
        target.with_suffix(".part").replace(target)
        LOG.info("fetched %s (%d utterances)", target.name, table.num_rows)
        return target

    def shards(self) -> list[Path]:
        """The local alignment shards, in name order."""
        return sorted(self.home.glob("*.parquet"))

    def metadata(self) -> Path:
        """The chapter table, author database and punctuated manifests, streamed out of their archives."""
        self.meta.mkdir(parents=True, exist_ok=True)
        for url, members in METADATA_ARCHIVES.items():
            with urlopen(url) as response, tarfile.open(fileobj=response, mode="r|gz") as archive:
                for member in archive:
                    if Path(member.name).name in members and (handle := archive.extractfile(member)):
                        (self.meta / Path(member.name).name).write_bytes(handle.read())
        return self.meta
