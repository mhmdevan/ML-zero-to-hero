from __future__ import annotations

from pathlib import Path

import pytest


def test_save_mnist_sample_main_writes_output(monkeypatch, tmp_path) -> None:
    torch = pytest.importorskip("torch")
    module = pytest.importorskip("src.save_mnist_sample")

    class FakeMNIST:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __getitem__(self, index: int):
            return torch.zeros((1, 28, 28), dtype=torch.float32), 5

    output_path = Path(tmp_path) / "sample.png"

    def fake_save_image(tensor, out_path):
        Path(out_path).write_bytes(b"fake")

    monkeypatch.setenv("FOUNDATIONS_MNIST_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("FOUNDATIONS_MNIST_SAMPLE_OUTPUT", str(output_path))
    monkeypatch.setattr(module.datasets, "MNIST", FakeMNIST)
    monkeypatch.setattr(module.utils, "save_image", fake_save_image)

    module.main()

    assert output_path.exists()
    assert output_path.read_bytes() == b"fake"
