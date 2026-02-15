from __future__ import annotations

import numpy as np

from src.numpy_basics import save_array_as_text


def test_save_array_as_text_writes_file(tmp_path) -> None:
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    out_path = tmp_path / "matrix.txt"

    save_array_as_text(arr, out_path, header="matrix")

    assert out_path.exists()
    text = out_path.read_text(encoding="utf-8")
    assert "matrix" in text
    assert "1.0000" in text
