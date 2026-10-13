import io
import tarfile
from pathlib import Path
from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from huggingface_hub import HfFileSystem

from core.paths import DataRoot
from megspeech.data.raw import librispeechfetch as module
from megspeech.data.raw.librispeechfetch import Fetch


class Remote:
    def __init__(self, table: pa.Table) -> None:
        buffer = io.BytesIO()
        pq.write_table(table.append_column("audio", pa.array([b"x"] * table.num_rows)), buffer)
        self.payload = buffer.getvalue()
        self.opened: list[str] = []

    def glob(self, path: str) -> list[str]:
        return [path.replace("*", "dev_clean-00000-of-00001"), path.replace("*", "dev_other-00000-of-00001")]

    def open(self, path: str, mode: str) -> io.BytesIO:
        self.opened.append(path)
        assert mode == "rb"
        return io.BytesIO(self.payload)


def remote() -> HfFileSystem:
    return cast(
        HfFileSystem,
        Remote(pa.table({"id": ["a"], "transcript": ["t"], "words": [[{"start": 0.0, "end": 1.0, "word": "t"}]]})),
    )


def test_alignments(tmp_path: Path) -> None:
    source = remote()
    paths = Fetch(DataRoot(tmp_path)).alignments(source)
    assert [p.name for p in paths] == ["dev_clean-00000-of-00001.parquet", "dev_other-00000-of-00001.parquet"]
    assert pq.read_table(paths[0]).column_names == ["id", "transcript", "words"]


def test_shard(tmp_path: Path) -> None:
    fetch = Fetch(DataRoot(tmp_path))
    fetch.home.mkdir(parents=True)
    source = Remote(pa.table({"id": ["a"], "transcript": ["t"], "words": [[{"start": 0.0, "end": 1.0, "word": "t"}]]}))
    fetch.shard(cast(HfFileSystem, source), "x/y/dev_clean-00000-of-00001.parquet")
    fetch.shard(cast(HfFileSystem, source), "x/y/dev_clean-00000-of-00001.parquet")
    assert len(source.opened) == 1
    assert not list(fetch.home.glob("*.part"))


def test_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def archive(names: list[str]) -> io.BytesIO:
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
            for name in names:
                data = name.encode()
                info = tarfile.TarInfo(name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
        buffer.seek(0)
        return buffer

    archives = {
        "raw-metadata": archive(["LibriSpeech/CHAPTERS.TXT", "LibriSpeech/raw_metadata/ia.sqlite.db"]),
        "manifests": archive(["dev-clean.json", "other.txt"]),
    }
    monkeypatch.setattr(module, "urlopen", lambda url: next(a for k, a in archives.items() if k in url))
    meta = Fetch(DataRoot(tmp_path)).metadata()
    assert sorted(p.name for p in meta.iterdir()) == ["CHAPTERS.TXT", "dev-clean.json"]


def test_shards(tmp_path: Path) -> None:
    fetch = Fetch(DataRoot(tmp_path))
    fetch.alignments(remote())
    assert [p.name for p in fetch.shards()] == ["dev_clean-00000-of-00001.parquet", "dev_other-00000-of-00001.parquet"]
