from __future__ import annotations

import json

import pytest

from src.benchmark_numpy_vs_torch_cpu import (
    BenchmarkConfig,
    benchmark_once,
    save_results,
)


def test_benchmark_once_numpy_returns_positive_runtime() -> None:
    cfg = BenchmarkConfig(matrix_size=64, repetitions=2, seed=0)
    result = benchmark_once("numpy", cfg)

    assert result.backend == "numpy"
    assert result.elapsed_seconds > 0.0
    assert result.avg_seconds_per_iter > 0.0


def test_benchmark_once_torch_returns_positive_runtime() -> None:
    pytest.importorskip("torch")

    cfg = BenchmarkConfig(matrix_size=64, repetitions=2, seed=0, torch_num_threads=1)
    result = benchmark_once("torch", cfg)

    assert result.backend == "torch"
    assert result.elapsed_seconds > 0.0
    assert result.avg_seconds_per_iter > 0.0


def test_save_results_writes_json_and_csv(tmp_path) -> None:
    cfg = BenchmarkConfig(matrix_size=32, repetitions=1, seed=1)
    results = [benchmark_once("numpy", cfg)]

    json_path, csv_path = save_results(results, tmp_path)

    assert json_path.exists()
    assert csv_path.exists()

    rows = json.loads(json_path.read_text(encoding="utf-8"))
    assert rows[0]["backend"] == "numpy"
