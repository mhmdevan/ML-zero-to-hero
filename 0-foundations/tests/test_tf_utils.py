from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from src.tf_lab.tf_utils import (
    confusion_matrix_np,
    ensure_dir,
    load_grayscale_28x28,
    project_root_from,
    quantiles_ms,
    read_json,
    write_csv,
    write_json,
)


@dataclass
class DummyConfig:
    x: int
    y: str


def test_write_and_read_json_roundtrip(tmp_path) -> None:
    path = tmp_path / "sample.json"
    write_json(path, DummyConfig(x=3, y="ok"))

    obj = read_json(path)
    assert obj["x"] == 3
    assert obj["y"] == "ok"


def test_write_csv_with_rows(tmp_path) -> None:
    path = tmp_path / "rows.csv"
    write_csv(path, rows=[{"a": 1, "b": 2}, {"a": 3, "b": 4}])

    text = path.read_text(encoding="utf-8")
    assert "a,b" in text
    assert "1,2" in text


def test_confusion_matrix_np_counts_correctly() -> None:
    y_true = np.array([0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 0, 2, 1])
    cm = confusion_matrix_np(y_true, y_pred, n_classes=3)

    assert cm.shape == (3, 3)
    assert cm[1, 0] == 1
    assert cm[2, 2] == 1


def test_load_grayscale_28x28(tmp_path) -> None:
    image_path = tmp_path / "img.png"
    Image.fromarray(np.ones((64, 64), dtype=np.uint8) * 200, mode="L").save(image_path)

    arr = load_grayscale_28x28(image_path)
    assert arr.shape == (28, 28)
    assert arr.dtype == np.float32
    assert float(arr.min()) >= 0.0 and float(arr.max()) <= 1.0


def test_quantiles_ms_has_expected_keys() -> None:
    metrics = quantiles_ms([0.01, 0.02, 0.03])
    assert set(metrics.keys()) == {"p50_ms", "p95_ms", "avg_ms"}


def test_project_root_from_and_ensure_dir(tmp_path) -> None:
    fake_file = tmp_path / "a" / "b" / "c.py"
    ensure_dir(fake_file.parent)
    fake_file.write_text("# x", encoding="utf-8")

    root = project_root_from(fake_file, up=2)
    assert root == tmp_path
