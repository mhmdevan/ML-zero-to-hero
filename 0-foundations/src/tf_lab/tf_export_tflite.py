#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import numpy as np
import tensorflow as tf

from .tf_utils import ensure_dir, project_root_from, seed_everything, set_env_quiet_tf, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export Keras/SavedModel to TFLite (optional quantization)")
    p.add_argument("--source-keras", type=str, default="models/tf_lab/mnist_mlp/model.keras", help="Path to .keras")
    p.add_argument("--source-savedmodel", type=str, default=None, help="Path to SavedModel dir (optional)")
    p.add_argument("--out", type=str, default="models/tf_lab/mnist_mlp/model.tflite", help="Output .tflite path")
    p.add_argument(
        "--quantize",
        type=str,
        default="none",
        choices=["none", "dynamic", "int8"],
        help="Quantization mode",
    )
    p.add_argument("--rep-samples", type=int, default=200, help="Representative samples for int8 quant")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_mnist_rep_data(n: int, seed: int) -> np.ndarray:
    (x_train, _), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype(np.float32) / 255.0
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x_train), size=min(n, len(x_train)), replace=False)
    return x_train[idx]


def rep_dataset_gen(samples: np.ndarray) -> Iterator[list[np.ndarray]]:
    # Converter expects an iterator yielding list of inputs
    for i in range(samples.shape[0]):
        x = samples[i : i + 1]  # [1,28,28]
        yield [x.astype(np.float32)]


def main() -> None:
    set_env_quiet_tf()
    args = parse_args()
    seed_everything(args.seed)

    PROJECT_ROOT = project_root_from(Path(__file__), up=2)
    out_path = PROJECT_ROOT / args.out
    ensure_dir(out_path.parent)

    if args.source_savedmodel:
        savedmodel_dir = PROJECT_ROOT / args.source_savedmodel
        if not savedmodel_dir.exists():
            raise FileNotFoundError(f"SavedModel not found: {savedmodel_dir}")
        converter = tf.lite.TFLiteConverter.from_saved_model(str(savedmodel_dir))
        source_kind = "savedmodel"
        source_path = str(savedmodel_dir)
    else:
        keras_path = PROJECT_ROOT / args.source_keras
        if not keras_path.exists():
            raise FileNotFoundError(f"Keras model not found: {keras_path}")
        model = tf.keras.models.load_model(str(keras_path))
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        source_kind = "keras"
        source_path = str(keras_path)

    if args.quantize == "dynamic":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    elif args.quantize == "int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        rep = load_mnist_rep_data(args.rep_samples, args.seed)
        converter.representative_dataset = lambda: rep_dataset_gen(rep)
        # full int8 (may require input/output types)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    out_path.write_bytes(tflite_model)
    print(f"[TFLITE] saved -> {out_path}")

    meta = {
        "source_kind": source_kind,
        "source_path": source_path,
        "quantize": args.quantize,
        "rep_samples": args.rep_samples if args.quantize == "int8" else 0,
        "task": "mnist_digit_classification",
        "input": {"shape": [1, 28, 28], "dtype": "float32 (or int8 if int8 quant)"},
        "output": {"shape": [1, 10], "type": "logits"},
        "notes": [
            "For int8 quant, input/output tensors are int8. You must scale/zero-point based on interpreter details.",
            "For none/dynamic, input is float32 normalized to [0,1].",
        ],
    }
    write_json(out_path.with_suffix(".meta.json"), meta)
    print(f"[META] saved -> {out_path.with_suffix('.meta.json')}")


if __name__ == "__main__":
    main()
