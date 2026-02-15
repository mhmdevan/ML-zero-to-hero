from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def project_root_from(file_path: Path, up: int = 2) -> Path:
    """
    file_path: typically __file__ under src/tf_lab/
    up=2 => /project/src/tf_lab/x.py -> parents[2] = /project
    """
    return file_path.resolve().parents[up]


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)

    if is_dataclass(obj) and not isinstance(obj, type):
        obj = asdict(obj)

    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text("utf-8"))


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    import csv

    ensure_dir(path.parent)
    if not rows:
        # create empty file with headers if provided
        with path.open("w", newline="", encoding="utf-8") as f:
            if fieldnames:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
        return

    if fieldnames is None:
        # keep stable ordering by first row
        fieldnames = list(rows[0].keys())

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def set_env_quiet_tf() -> None:
    # Reduce TF C++ logs
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def seed_everything(seed: int = 42) -> None:
    import random

    import tensorflow as tf

    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

    # Deterministic ops can slow down; enable only if you really want strict determinism.
    # In some TF builds this may raise if not supported; ignore safely.
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def plot_training_curves(
    out_path: Path,
    history: Dict[str, List[float]],
    title: str = "Training Curves",
) -> None:
    """
    history: dict like {"loss":[...], "val_loss":[...], "accuracy":[...], "val_accuracy":[...]}
    Saves a single figure with available curves.
    """
    import matplotlib.pyplot as plt

    ensure_dir(out_path.parent)

    keys = list(history.keys())
    if not keys:
        return

    # Pick common pairs if exist
    plt.figure(figsize=(10, 6))
    for k in keys:
        plt.plot(history[k], label=k)

    plt.title(title)
    plt.xlabel("epoch")
    plt.ylabel("value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def confusion_matrix_np(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def plot_confusion_matrix(
    out_path: Path,
    cm: np.ndarray,
    class_names: Sequence[str],
    title: str = "Confusion Matrix",
) -> None:
    import matplotlib.pyplot as plt

    ensure_dir(out_path.parent)

    plt.figure(figsize=(8, 7))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    # annotate
    thresh = cm.max() * 0.5 if cm.max() > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cm[i, j]
            plt.text(
                j,
                i,
                str(v),
                ha="center",
                va="center",
                color="white" if v > thresh else "black",
                fontsize=9,
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def load_grayscale_28x28(image_path: Path) -> np.ndarray:
    """
    Load a custom image and convert it to MNIST-like format:
      - grayscale
      - resize 28x28
      - float32 in [0,1]
      - shape (28,28)
    """
    from PIL import Image

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = Image.open(image_path).convert("L")
    img = img.resize((28, 28))
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr


def quantiles_ms(values_sec: List[float]) -> Dict[str, float]:
    x = np.array(values_sec, dtype=float)
    return {
        "p50_ms": float(np.quantile(x, 0.50) * 1000.0),
        "p95_ms": float(np.quantile(x, 0.95) * 1000.0),
        "avg_ms": float(np.mean(x) * 1000.0),
    }
