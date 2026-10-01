from pathlib import Path

import pytest

from core.paths import COMPETITION, ENV_OVERRIDE, DataRoot


@pytest.fixture
def config(tmp_path: Path) -> Path:
    written = tmp_path / "paths.yaml"
    written.write_text(f"data: {(tmp_path / 'root').as_posix()}\n", encoding="utf-8")
    return written


def test_from_config(config: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(ENV_OVERRIDE, raising=False)
    assert DataRoot.from_config(config).raw("x") == tmp_path / "root" / "raw" / "x"


def test_from_config_prefers_the_env_override(config: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A runner without the dataset points the override at an empty dir so data-gated tests skip."""
    monkeypatch.setenv(ENV_OVERRIDE, str(tmp_path / "elsewhere"))
    assert DataRoot.from_config(config).raw("x") == tmp_path / "elsewhere" / "raw" / "x"


def test_from_config_reads_nothing_when_the_env_override_is_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The override is an escape hatch for a runner with no paths.yaml, so it cannot require one."""
    monkeypatch.setenv(ENV_OVERRIDE, str(tmp_path / "elsewhere"))
    assert DataRoot.from_config(tmp_path / "absent.yaml").raw("x") == tmp_path / "elsewhere" / "raw" / "x"


def test_from_config_carries_a_per_kind_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(ENV_OVERRIDE, raising=False)
    written = tmp_path / "paths.yaml"
    written.write_text(
        f"data: {(tmp_path / 'root').as_posix()}\nprocessed: {(tmp_path / 'scratch').as_posix()}\n",
        encoding="utf-8",
    )
    assert DataRoot.from_config(written).processed("x") == tmp_path / "scratch" / "x"


def test_kind(tmp_path: Path):
    assert DataRoot(tmp_path).kind("reference") == tmp_path / "reference"


def test_kind_prefers_a_stated_override(tmp_path: Path):
    """A kind named in paths.yaml moves that kind alone — raw/ can stay on the slow disk."""
    stated = (tmp_path / "elsewhere").as_posix()
    assert DataRoot(tmp_path, {"processed": stated}).kind("processed") == Path(stated)


def test_raw(tmp_path: Path):
    """`raw` never creates anything — the download is read-only."""
    assert not DataRoot(tmp_path).raw("dataset").exists()


def test_raw_defaults_to_the_competition(tmp_path: Path):
    assert DataRoot(tmp_path).raw() == tmp_path / "raw" / COMPETITION


def test_processed(tmp_path: Path):
    """Derived artefacts are written, so the directory has to exist by the time the caller has the path."""
    assert DataRoot(tmp_path).processed("dataset").is_dir()


def test_models(tmp_path: Path):
    """Trained weights are written, and they are NOT under the processed root."""
    root = DataRoot(tmp_path)
    assert root.models("dataset").is_dir()
    assert root.models("dataset") != root.processed("dataset")


def test_models_takes_a_paths_yaml_redirect(tmp_path: Path):
    stated = (tmp_path / "elsewhere").as_posix()
    assert DataRoot(tmp_path, {"models": stated}).models("dataset") == Path(stated) / "dataset"


def test_foreign(tmp_path: Path):
    """Third-party weights sit under models/foreign/, apart from our own checkpoints."""
    root = DataRoot(tmp_path)
    assert root.foreign("MEG-XL") == tmp_path / "models" / "foreign" / "MEG-XL"
    assert root.foreign("MEG-XL").is_dir()
