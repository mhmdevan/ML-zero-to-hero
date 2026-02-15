from __future__ import annotations

from pathlib import Path


def test_source_module_inventory_stays_explicit() -> None:
    project_root = Path(__file__).resolve().parents[1]

    actual = {
        str(path.relative_to(project_root))
        for path in (project_root / "src").rglob("*.py")
    }

    expected = {
        "src/__init__.py",
        "src/benchmark_numpy_vs_torch_cpu.py",
        "src/gradient_descent_demo.py",
        "src/linear_algebra_demo.py",
        "src/numerical_stability.py",
        "src/numpy_basics.py",
        "src/numpy_manual_stats.py",
        "src/save_mnist_sample.py",
        "src/stats_basics.py",
        "src/tf_lab/tf_export_tflite.py",
        "src/tf_lab/tf_gradient_tape_demo.py",
        "src/tf_lab/tf_mnist_inference.py",
        "src/tf_lab/tf_mnist_mlp_train.py",
        "src/tf_lab/tf_timeseries_lstm.py",
        "src/tf_lab/tf_utils.py",
        "src/torch_autograd_gd.py",
        "src/torch_lstm_timeseries.py",
        "src/torch_mlp_mnist.py",
        "src/torch_mnist_inference.py",
    }

    assert actual == expected
