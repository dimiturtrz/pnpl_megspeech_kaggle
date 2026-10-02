import sys
from fnmatch import fnmatch
from pathlib import Path

import pytest

from core.paths import ENV_OVERRIDE
from megspeech import fetch as module
from megspeech.fetch import HOLDOUT, OTHERS, SOURCES, Source, main

Calls = list[dict[str, object]]


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> Calls:
    recorded: Calls = []

    def snapshot_download(**kwargs: object) -> str:
        recorded.append(kwargs)
        return str(kwargs["local_dir"])

    monkeypatch.setattr(module, "snapshot_download", snapshot_download)
    return recorded


def test_fetch(calls: Calls, tmp_path: Path):
    """The repo-relative tree lands under the destination, anonymously — the pnpl repos are public."""
    source = Source("org/repo", ("*.npz",))
    assert source.fetch(tmp_path) == tmp_path
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
