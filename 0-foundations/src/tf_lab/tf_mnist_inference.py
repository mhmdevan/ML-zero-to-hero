#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import tensorflow as tf

from .tf_utils import load_grayscale_28x28, project_root_from, read_json, seed_everything, set_env_quiet_tf


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MNIST inference (Keras/SavedModel/TFLite)")
    p.add_argument("--keras", type=str, default="models/tf_lab/mnist_mlp/model.keras", help="Path to .keras model")
    p.add_argument("--savedmodel", type=str, default=None, help="Path to SavedModel directory (optional)")
    p.add_argument("--tflite", type=str, default=None, help="Path to .tflite file (optional)")
    p.add_argument("--index", type=int, default=None, help="MNIST test index (0..9999)")
    p.add_argument("--image-path", type=str, default=None, help="Path to custom image")
    p.add_argument("--topk", type=int, default=3, help="Top-k predictions to print")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_mnist_test() -> tuple[np.ndarray, np.ndarray]:
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = x_test.astype(np.float32) / 255.0
    y_test = y_test.astype(np.int64)
    return x_test, y_test


def preprocess_single(img_28x28: np.ndarray) -> np.ndarray:
    if img_28x28.shape != (28, 28):
        raise ValueError(f"Expected (28,28) got {img_28x28.shape}")
    x = img_28x28.astype(np.float32)
    x = np.expand_dims(x, axis=0)  # [1,28,28]
    return x


def predict_keras(model: tf.keras.Model, x: np.ndarray) -> Dict[str, Any]:
    logits = model.predict(x, verbose=0)
    probs = tf.nn.softmax(logits, axis=1).numpy()[0]
    pred = int(np.argmax(probs))
    return {"pred": pred, "probs": probs.tolist()}


def predict_tflite(tflite_path: Path, x: np.ndarray) -> Dict[str, Any]:
    if not tflite_path.exists():
        raise FileNotFoundError(f"TFLite not found: {tflite_path}")

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Most MNIST models: input shape [1,28,28] or [1,28,28,1]
    inp = input_details[0]
    in_shape = inp["shape"]

    x_in = x
    if len(in_shape) == 4:
        # [1,28,28,1]
        x_in = np.expand_dims(x, axis=-1)

    # Ensure dtype
    x_in = x_in.astype(inp["dtype"])
    interpreter.set_tensor(inp["index"], x_in)
    interpreter.invoke()

    out = interpreter.get_tensor(output_details[0]["index"])
    # out could be logits
    out = out.reshape(-1)
    probs = tf.nn.softmax(out).numpy()
    pred = int(np.argmax(probs))
    return {"pred": pred, "probs": probs.tolist()}


def main() -> None:
    set_env_quiet_tf()
    args = parse_args()
    seed_everything(args.seed)

    PROJECT_ROOT = project_root_from(Path(__file__), up=2)

    # Decide input source
    if args.image_path:
        img = load_grayscale_28x28(Path(args.image_path))
        true_label: Optional[int] = None
        source = f"custom:{args.image_path}"
    else:
        x_test, y_test = load_mnist_test()
        idx = args.index if args.index is not None else 42
        if idx < 0 or idx >= len(x_test):
            raise IndexError(f"index out of range: {idx}")
        img = x_test[idx]
        true_label = int(y_test[idx])
        source = f"mnist_test_index:{idx}"

    x = preprocess_single(img)

    # Choose backend
    if args.tflite:
        res = predict_tflite(PROJECT_ROOT / args.tflite, x)
        backend = "tflite"
    elif args.savedmodel:
        model = tf.saved_model.load(str(PROJECT_ROOT / args.savedmodel))
        # SavedModel from Keras export typically requires calling signature
        # But simplest: prefer .keras for inference; if you insist on SavedModel:
        # you can export a serving function or use keras .keras.
        # So here, we fallback to .keras if savedmodel load isn't callable.
        try:
            infer = model.signatures["serving_default"]
            # find input key
            input_key = list(infer.structured_input_signature[1].keys())[0]
            out = infer(tf.convert_to_tensor(x))[list(infer.structured_outputs.keys())[0]].numpy()
            probs = tf.nn.softmax(out, axis=1).numpy()[0]
            res = {"pred": int(np.argmax(probs)), "probs": probs.tolist()}
            backend = "savedmodel_signature"
        except Exception:
            # fallback to keras
            model = tf.keras.models.load_model(str(PROJECT_ROOT / args.keras))
            res = predict_keras(model, x)
            backend = "keras_fallback"
    else:
        model = tf.keras.models.load_model(str(PROJECT_ROOT / args.keras))
        res = predict_keras(model, x)
        backend = "keras"

    probs = np.array(res["probs"], dtype=float)
    topk = min(args.topk, probs.size)
    top_idx = probs.argsort()[::-1][:topk]

    print("\n=== MNIST Inference ===")
    print(f"source: {source}")
    if true_label is not None:
        print(f"true_label: {true_label}")
    print(f"backend: {backend}")
    print(f"pred: {res['pred']}")
    print("topk:")
    for i in top_idx:
        print(f"  digit={int(i)} prob={probs[i]:.6f}")


if __name__ == "__main__":
    main()
