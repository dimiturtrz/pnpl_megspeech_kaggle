import sys
from fnmatch import fnmatch
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pytest

from core.paths import ENV_OVERRIDE, DataRoot
from megspeech.data.raw import fetch as module
from megspeech.data.raw.fetch import (
    HOLDOUT,
    MEGXL,
    NEUROMAG,
    OTHERS,
    PINNED,
    SOURCES,
    WEIGHTS,
    Checkout,
    Download,
    HubRepo,
    Weights,
    main,
)

Calls = list[dict[str, object]]


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> Calls:
    recorded: Calls = []

    def snapshot_download(**kwargs: object) -> str:
        recorded.append(kwargs)
        return str(kwargs["local_dir"])

    monkeypatch.setattr(module, "snapshot_download", snapshot_download)
    return recorded


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DataRoot:
    monkeypatch.setenv(ENV_OVERRIDE, str(tmp_path))
    return DataRoot.from_config()


def test_hub_repo_fetch(calls: Calls, root: DataRoot):
    """The repo-relative tree lands under the destination, anonymously — the pnpl repos are public."""
    source = HubRepo("org/repo", ("*.npz",))
    assert source.fetch(root) == root.raw()
    assert calls[0]["repo_id"] == "org/repo"
    assert calls[0]["repo_type"] == "dataset"
    assert calls[0]["allow_patterns"] == ["*.npz"]
    assert calls[0]["token"] is False


def test_sources_never_pull_raw_fif():
    assert all(".fif" not in pattern for source in SOURCES for pattern in source.patterns)


def test_main_fetches_the_holdout_first(calls: Calls, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(ENV_OVERRIDE, str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["fetch"])
    main()
    assert [call["repo_id"] for call in calls] == [source.repo for source in SOURCES]
    assert calls[0]["repo_id"] == HOLDOUT.repo
    assert calls[0]["local_dir"] == tmp_path / "raw" / "pnpl_libribrain"


def test_main_holdout_only(calls: Calls, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(ENV_OVERRIDE, str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["fetch", "--holdout-only"])
    main()
    assert [call["repo_id"] for call in calls] == [HOLDOUT.repo]


def test_main_others(calls: Calls, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(ENV_OVERRIDE, str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["fetch", "--others"])
    main()
    assert [call["allow_patterns"] for call in calls] == [list(OTHERS.patterns)]


def test_others_exclude_subject0():
    names = [
        "Sherlock1/derivatives/serialised/sub-0_ses-1_task-Sherlock1_run-1_meg.h5",
        "x/sub-12_ses-1_task-Sherlock1_run-1_meg.h5",
        "x/sub-3_ses-2_events.tsv",
    ]
    assert [any(fnmatch(n, p) for p in OTHERS.patterns) for n in names] == [False, True, True]


def test_hub_repo_home(root: DataRoot):
    assert HubRepo("org/data", ("*",)).home(root) == root.raw()


def test_weights_home(root: DataRoot):
    assert Weights("org/net", ("*",)).home(root) == root.foreign("net")


def test_main_weights(calls: Calls, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(ENV_OVERRIDE, str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["fetch", "--weights"])
    fetched: list[object] = []
    monkeypatch.setattr(Checkout, "fetch", lambda self, _root: fetched.append(self))
    monkeypatch.setattr(Download, "fetch", lambda self, _root: fetched.append(self))
    main()
    assert [call["repo_id"] for call in calls] == [source.repo for source in WEIGHTS]
    assert {call["repo_type"] for call in calls} == {"model"}
    assert fetched == [MEGXL, NEUROMAG]


def zipped(files: dict[str, bytes]) -> bytes:
    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return buffer.getvalue()


@pytest.fixture
def served(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Every URL opened, each answered with one archive whose tree sits under a top-level folder."""
    urls: list[str] = []
    payload = zipped({"repo-abc/pkg/mod.py": b"x = 1", "repo-abc/README.md": b"hi"})

    def urlopen(url: str) -> BytesIO:
        urls.append(url)
        return BytesIO(payload)

    monkeypatch.setattr(module, "urlopen", urlopen)
    return urls


def test_checkout_home(tmp_path: Path):
    assert Checkout("org/Repo", "abc", tmp_path).home() == tmp_path / "Repo"


def test_checkout_archive():
    assert Checkout("org/repo", "abc").archive() == "https://github.com/org/repo/archive/abc.zip"


def test_checkout_fetch(served: list[str], tmp_path: Path, root: DataRoot):
    checkout = Checkout("org/repo", "abc", tmp_path)
    home = checkout.fetch(root)
    assert (home / "pkg" / "mod.py").read_bytes() == b"x = 1"
    assert (home / PINNED).read_text() == "abc"
    assert checkout.fetch(root) == home
    assert served == [checkout.archive()]
    Checkout("org/repo", "def", tmp_path).fetch(root)
    assert len(served) == 2


def test_download_home(root: DataRoot):
    assert Download("https://host/a/file.fif", "set").home(root) == root.raw("set") / "file.fif"


def test_download_fetch(served: list[str], root: DataRoot):
    download = Download("https://host/a/file.zip", "set")
    home = download.fetch(root)
    assert home.read_bytes().startswith(b"PK")
    assert download.fetch(root) == home
    assert served == [download.url]
