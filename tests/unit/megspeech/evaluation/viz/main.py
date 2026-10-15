import argparse
from pathlib import Path

import pytest

import megspeech.evaluation.viz.__main__ as entry
from megspeech.device import Device
from megspeech.evaluation.viz.__main__ import Cli
from megspeech.evaluation.viz.export import ExportConfig, Sampler
from megspeech.evaluation.viz.outputs import Reader
from megspeech.evaluation.viz.serve import ServeConfig


class FakeExporter:
    seen: list[ExportConfig] = []  # noqa: RUF012
    devices: list[str] = []  # noqa: RUF012

    def __init__(self, sampler: Sampler, reader: Reader) -> None:
        self.devices += [sampler.device, reader.device]

    def run(self, config: ExportConfig) -> Path:
        self.seen.append(config)
        return Path(config.name)


class FakeServer:
    seen: list[ServeConfig] = []  # noqa: RUF012

    def __init__(self, config: ServeConfig) -> None:
        self.seen.append(config)

    def serve(self) -> None:
        raise KeyboardInterrupt


@pytest.fixture
def fakes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MEGSPEECH_DATA", str(tmp_path))
    monkeypatch.setattr(entry, "Exporter", FakeExporter)
    monkeypatch.setattr(entry, "Server", FakeServer)
    FakeExporter.seen.clear()
    FakeExporter.devices.clear()
    FakeServer.seen.clear()


def test_parser() -> None:
    args = Cli().parser().parse_args(["export", "--models", "a", "b", "--n", "5"])
    assert (args.models, args.n, args.seed, args.name) == (["a", "b"], 5, 0, "default")
    assert Cli().parser().parse_args(["serve"]).port == 8000
    with pytest.raises(SystemExit):
        Cli().parser().parse_args([])


@pytest.mark.usefixtures("fakes")
def test_export() -> None:
    Cli().export(argparse.Namespace(models=["a"], n=3, seed=1, name="x", device=Device.CUDA))
    assert FakeExporter.seen == [ExportConfig(models=("a",), n=3, seed=1, name="x")]
    assert FakeExporter.devices == [Device.CUDA, Device.CUDA]


@pytest.mark.usefixtures("fakes")
def test_serve(tmp_path: Path) -> None:
    Cli().serve(argparse.Namespace(port=0, name="x"))
    assert FakeServer.seen == [ServeConfig(data=tmp_path / "viz" / "x", port=0)]


@pytest.mark.usefixtures("fakes")
def test_run(tmp_path: Path) -> None:
    Cli().run(["serve", "--port", "9001", "--name", "y"])
    Cli().run(["export", "--models", "m"])
    assert FakeServer.seen == [ServeConfig(data=tmp_path / "viz" / "y", port=9001)]
    assert FakeExporter.seen == [ExportConfig(models=("m",))]
    assert FakeExporter.devices == [Device.CPU, Device.CPU]
