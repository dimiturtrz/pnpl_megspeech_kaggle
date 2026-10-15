import pytest
import torch

from megspeech.device import Device


def test_best(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert Device.best() is Device.CPU
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert Device.best() is Device.CUDA


def test_as_torch() -> None:
    assert Device.CPU.as_torch() == torch.device("cpu")
    assert Device.CUDA.as_torch().type == Device.CUDA


def test_autocast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with Device.autocast():
        assert not torch.is_autocast_enabled(Device.CUDA)
