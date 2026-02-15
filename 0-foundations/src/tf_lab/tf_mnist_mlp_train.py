#!/usr/bin/env python
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import tensorflow as tf

from .tf_utils import (
    confusion_matrix_np,
    ensure_dir,
    plot_confusion_matrix,
    plot_training_curves,
    project_root_from,
    seed_everything,
    set_env_quiet_tf,
    write_csv,
    write_json,
)


@dataclass
class MNISTConfig:
    seed: int = 42

    # Data
    val_ratio: float = 0.1
    batch_size: int = 128
    shuffle_buffer: int = 10_000

    # Model
    hidden_dims: Tuple[int, int] = (256, 128)
    lr: float = 1e-3
    epochs: int = 8
    early_stop_patience: int = 2

    # Paths (filled in runtime)
    data_dir: Path | None = None
    out_dir: Path | None = None
    model_dir: Path | None = None


def load_mnist_np() -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # normalize
    x_train = (x_train.astype(np.float32) / 255.0)
    x_test = (x_test.astype(np.float32) / 255.0)
    return (x_train, y_train.astype(np.int64)), (x_test, y_test.astype(np.int64))


def split_train_val(
    x: np.ndarray,
    y: np.ndarray,
    val_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = x.shape[0]
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    val_size = int(n * val_ratio)
    val_idx = idx[:val_size]
    tr_idx = idx[val_size:]

    return x[tr_idx], y[tr_idx], x[val_idx], y[val_idx]


def make_tf_dataset(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    shuffle_buffer: int,
    seed: int,
) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(shuffle_buffer, seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def build_mlp(cfg: MNISTConfig) -> tf.keras.Model:
    h1, h2 = cfg.hidden_dims
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(28, 28)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(h1, activation="relu"),
            tf.keras.layers.Dense(h2, activation="relu"),
            tf.keras.layers.Dense(10),  # logits
        ]
    )
    return model


def export_savedmodel(model: tf.keras.Model, out_dir: Path) -> Path:
    """
    Keras 3 / TF newer versions:
      - model.export(dir) is preferred for SavedModel export
    Older:
      - model.save(dir) without extension will produce SavedModel
    We try export() then fallback.
    """
    ensure_dir(out_dir)
    try:
        # Keras 3 style
        model.export(str(out_dir))
    except Exception:
        # fallback: Keras will treat directory as SavedModel target
        model.save(str(out_dir), include_optimizer=False)
    return out_dir


def main() -> None:
    set_env_quiet_tf()

    cfg = MNISTConfig()
    seed_everything(cfg.seed)

    PROJECT_ROOT = project_root_from(Path(__file__), up=2)
    cfg.data_dir = PROJECT_ROOT / "data" / "mnist"
    cfg.out_dir = PROJECT_ROOT / "output" / "tf_lab" / "mnist_mlp"
    cfg.model_dir = PROJECT_ROOT / "models" / "tf_lab" / "mnist_mlp"

    ensure_dir(cfg.data_dir)
    ensure_dir(cfg.out_dir)
    ensure_dir(cfg.model_dir)

    print(f"[TF] device(s): {tf.config.list_physical_devices()}")
    print(f"[PATH] out_dir={cfg.out_dir}")
    print(f"[PATH] model_dir={cfg.model_dir}")

    # -----------------------
    # Load data
    # -----------------------
    (x_train_full, y_train_full), (x_test, y_test) = load_mnist_np()
    x_train, y_train, x_val, y_val = split_train_val(x_train_full, y_train_full, cfg.val_ratio, cfg.seed)

    print(f"[DATA] train={x_train.shape} val={x_val.shape} test={x_test.shape}")

    train_ds = make_tf_dataset(x_train, y_train, cfg.batch_size, True, cfg.shuffle_buffer, cfg.seed)
    val_ds = make_tf_dataset(x_val, y_val, cfg.batch_size, False, cfg.shuffle_buffer, cfg.seed)
    test_ds = make_tf_dataset(x_test, y_test, cfg.batch_size, False, cfg.shuffle_buffer, cfg.seed)

    # -----------------------
    # Model
    # -----------------------
    model = build_mlp(cfg)
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )

    # -----------------------
    # Callbacks
    # -----------------------
    tb_dir = cfg.out_dir / "tensorboard"
    ensure_dir(tb_dir)

    ckpt_keras = cfg.model_dir / "model.keras"
    callbacks: List[tf.keras.callbacks.Callback] = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_acc",
            patience=cfg.early_stop_patience,
            mode="max",
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_keras),
            monitor="val_acc",
            mode="max",
            save_best_only=True,
        ),
        tf.keras.callbacks.TensorBoard(log_dir=str(tb_dir)),
        tf.keras.callbacks.CSVLogger(str(cfg.out_dir / "history.csv")),
    ]

    # -----------------------
    # Train
    # -----------------------
    t0 = time.perf_counter()
    hist = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.epochs,
        callbacks=callbacks,
        verbose=1,
    )
    t1 = time.perf_counter()

    # -----------------------
    # Evaluate
    # -----------------------
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"[TEST] loss={test_loss:.4f} acc={test_acc*100:.2f}%")
    train_time_sec = float(t1 - t0)

    # -----------------------
    # Save artifacts
    # -----------------------
    # 1) Save best keras (already checkpointed) + final keras
    final_keras = cfg.model_dir / "final_model.keras"
    model.save(str(final_keras))
    print(f"[SAVE] final keras -> {final_keras}")
    print(f"[SAVE] best keras  -> {ckpt_keras}")

    # 2) Export SavedModel
    savedmodel_dir = cfg.model_dir / "savedmodel"
    export_savedmodel(model, savedmodel_dir)
    print(f"[EXPORT] savedmodel -> {savedmodel_dir}")

    # 3) Metrics JSON
    history_dict: Dict[str, Any] = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
    metrics = {
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "train_time_sec": train_time_sec,
        "config": {
            "seed": cfg.seed,
            "val_ratio": cfg.val_ratio,
            "batch_size": cfg.batch_size,
            "hidden_dims": list(cfg.hidden_dims),
            "lr": cfg.lr,
            "epochs": cfg.epochs,
        },
        "history": history_dict,
        "paths": {
            "best_keras": str(ckpt_keras),
            "final_keras": str(final_keras),
            "savedmodel": str(savedmodel_dir),
            "tensorboard": str(tb_dir),
        },
    }
    write_json(cfg.out_dir / "metrics.json", metrics)

    # 4) Plot curves
    plot_training_curves(cfg.out_dir / "training_curves.png", history_dict, title="MNIST MLP Training Curves")

    # 5) Confusion matrix on test
    #    We'll predict in numpy for full test set (60k? actually test is 10k)
    logits = model.predict(x_test, batch_size=cfg.batch_size, verbose=0)
    y_pred = np.argmax(logits, axis=1)

    cm = confusion_matrix_np(y_test, y_pred, n_classes=10)
    class_names = [str(i) for i in range(10)]
    plot_confusion_matrix(cfg.out_dir / "confusion_matrix.png", cm, class_names, title="MNIST Confusion Matrix")

    write_json(cfg.out_dir / "confusion_matrix.json", {"cm": cm.tolist(), "class_names": class_names})

    print(f"[DONE] metrics -> {cfg.out_dir / 'metrics.json'}")
    print(f"[DONE] plots   -> {cfg.out_dir / 'training_curves.png'} , {cfg.out_dir / 'confusion_matrix.png'}")
    print(f"[DONE] tb logs -> {tb_dir}")
    print("\nRun TensorBoard:")
    print(f"  tensorboard --logdir {tb_dir}")


if __name__ == "__main__":
    main()
