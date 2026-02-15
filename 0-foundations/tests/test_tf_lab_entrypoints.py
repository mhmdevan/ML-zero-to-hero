from __future__ import annotations

import sys

import pytest


def test_tf_lab_modules_import_and_parse_args(monkeypatch) -> None:
    pytest.importorskip("tensorflow")

    tf_gradient = pytest.importorskip("src.tf_lab.tf_gradient_tape_demo")
    tf_train = pytest.importorskip("src.tf_lab.tf_mnist_mlp_train")
    tf_infer = pytest.importorskip("src.tf_lab.tf_mnist_inference")
    tf_export = pytest.importorskip("src.tf_lab.tf_export_tflite")
    tf_series = pytest.importorskip("src.tf_lab.tf_timeseries_lstm")

    assert callable(tf_gradient.main)

    monkeypatch.setattr(sys, "argv", ["prog"])
    _ = tf_train.parse_args()
    _ = tf_infer.parse_args()
    _ = tf_export.parse_args()
    _ = tf_series.parse_args()
