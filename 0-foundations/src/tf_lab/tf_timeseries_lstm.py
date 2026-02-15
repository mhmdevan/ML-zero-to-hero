#!/usr/bin/env python
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import tensorflow as tf

from .tf_utils import (
    ensure_dir,
    plot_training_curves,
    project_root_from,
    seed_everything,
    set_env_quiet_tf,
    write_json,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Time-series forecasting with TF LSTM + tf.data windowing")
    p.add_argument("--series-len", type=int, default=6000, help="Length of synthetic series")
    p.add_argument("--window", type=int, default=64, help="Window length")
    p.add_argument("--horizon", type=int, default=1, help="Forecast horizon (steps ahead)")
    p.add_argument("--batch", type=int, default=128, help="Batch size")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def make_sine_series(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float32)
    # mixture + noise
    s = (
        0.6 * np.sin(2 * np.pi * t / 50.0)
        + 0.3 * np.sin(2 * np.pi * t / 200.0)
        + 0.1 * np.sin(2 * np.pi * t / 15.0)
    )
    noise = rng.normal(0.0, 0.15, size=n).astype(np.float32)
    y = (s + noise).astype(np.float32)
    return y


def train_val_test_split(series: np.ndarray, train_ratio=0.7, val_ratio=0.15):
    n = len(series)
    tr_end = int(n * train_ratio)
    va_end = int(n * (train_ratio + val_ratio))
    return series[:tr_end], series[tr_end:va_end], series[va_end:]


def make_window_ds(
    series: np.ndarray,
    window: int,
    horizon: int,
    batch: int,
    shuffle: bool,
    seed: int,
) -> tf.data.Dataset:
    """
    Build dataset of (x, y):
      x: [window, 1]
      y: [horizon] or scalar if horizon=1
    """
    s = tf.convert_to_tensor(series, dtype=tf.float32)
    s = tf.expand_dims(s, axis=-1)  # [N,1]

    total_window = window + horizon
    ds = tf.data.Dataset.from_tensor_slices(s)
    ds = ds.window(total_window, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(total_window))
    if shuffle:
        ds = ds.shuffle(10_000, seed=seed, reshuffle_each_iteration=True)

    def split_xy(w: tf.Tensor):
        x = w[:window]                 # [window,1]
        y = w[window:]                 # [horizon,1]
        y = tf.squeeze(y, axis=-1)     # [horizon]
        return x, y

    ds = ds.map(split_xy, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch).prefetch(tf.data.AUTOTUNE)
    return ds


def baseline_last_value(x: np.ndarray, horizon: int) -> np.ndarray:
    """
    Baseline: predict the last observed value repeated horizon times.
    x shape: [num_samples, window]
    """
    last = x[:, -1]  # [n]
    if horizon == 1:
        return last.reshape(-1, 1)
    return np.repeat(last.reshape(-1, 1), repeats=horizon, axis=1)


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def build_lstm(window: int, horizon: int) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(window, 1)),
            tf.keras.layers.LSTM(64),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(horizon),
        ]
    )
    return model


def export_savedmodel(model: tf.keras.Model, out_dir: Path) -> Path:
    ensure_dir(out_dir)
    try:
        model.export(str(out_dir))
    except Exception:
        model.save(str(out_dir), include_optimizer=False)
    return out_dir


def main() -> None:
    set_env_quiet_tf()
    args = parse_args()
    seed_everything(args.seed)

    PROJECT_ROOT = project_root_from(Path(__file__), up=2)
    out_dir = PROJECT_ROOT / "output" / "tf_lab" / "timeseries_lstm"
    model_dir = PROJECT_ROOT / "models" / "tf_lab" / "timeseries_lstm"
    ensure_dir(out_dir)
    ensure_dir(model_dir)

    # -------------------
    # Data
    # -------------------
    series = make_sine_series(args.series_len, args.seed)
    tr, va, te = train_val_test_split(series)

    train_ds = make_window_ds(tr, args.window, args.horizon, args.batch, shuffle=True, seed=args.seed)
    val_ds = make_window_ds(va, args.window, args.horizon, args.batch, shuffle=False, seed=args.seed)
    test_ds = make_window_ds(te, args.window, args.horizon, args.batch, shuffle=False, seed=args.seed)

    print(f"[DATA] len(series)={len(series)} train={len(tr)} val={len(va)} test={len(te)}")
    print(f"[CFG] window={args.window} horizon={args.horizon} batch={args.batch}")

    # -------------------
    # Model
    # -------------------
    model = build_lstm(args.window, args.horizon)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss="mse",
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_mae", patience=2, mode="min", restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_dir / "model.keras"),
            monitor="val_mae",
            mode="min",
            save_best_only=True,
        ),
    ]

    # -------------------
    # Train
    # -------------------
    t0 = time.perf_counter()
    hist = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )
    t1 = time.perf_counter()
    train_time = float(t1 - t0)

    # -------------------
    # Evaluate (small)
    # -------------------
    test_loss, test_mae = model.evaluate(test_ds, verbose=0)
    print(f"[TEST] loss(mse)={test_loss:.6f} mae={test_mae:.6f}")

    # -------------------
    # Baseline comparison on a slice (lightweight, no huge RAM)
    # -------------------
    # We'll build a numpy sample of windows from test part for baseline vs model.
    # Limit samples to avoid memory spikes.
    max_samples = 5000
    s = te
    n_possible = len(s) - (args.window + args.horizon) + 1
    n = min(max_samples, max(0, n_possible))

    xs = np.zeros((n, args.window), dtype=np.float32)
    ys = np.zeros((n, args.horizon), dtype=np.float32)
    for i in range(n):
        chunk = s[i : i + args.window + args.horizon]
        xs[i] = chunk[: args.window]
        ys[i] = chunk[args.window :]

    baseline_pred = baseline_last_value(xs, args.horizon)  # [n,horizon]
    # model expects [n,window,1]
    model_pred = model.predict(xs[..., None], batch_size=1024, verbose=0)

    baseline_mae = compute_mae(ys, baseline_pred)
    baseline_mse = compute_mse(ys, baseline_pred)
    model_mae = compute_mae(ys, model_pred)
    model_mse = compute_mse(ys, model_pred)

    print(f"[BASELINE] last-value | mae={baseline_mae:.6f} mse={baseline_mse:.6f}")
    print(f"[MODEL]    lstm       | mae={model_mae:.6f} mse={model_mse:.6f}")

    # -------------------
    # Save artifacts
    # -------------------
    history_dict: Dict[str, List[float]] = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
    plot_training_curves(out_dir / "training_curves.png", history_dict, title="Time-series LSTM Training Curves")

    savedmodel_dir = export_savedmodel(model, model_dir / "savedmodel")
    final_keras = model_dir / "final_model.keras"
    model.save(str(final_keras))

    metrics = {
        "config": {
            "series_len": args.series_len,
            "window": args.window,
            "horizon": args.horizon,
            "batch": args.batch,
            "epochs": args.epochs,
            "lr": args.lr,
            "seed": args.seed,
        },
        "train_time_sec": train_time,
        "test": {"mse": float(test_loss), "mae": float(test_mae)},
        "baseline_vs_model_sampled": {
            "n_samples": int(n),
            "baseline": {"mae": baseline_mae, "mse": baseline_mse},
            "model": {"mae": model_mae, "mse": model_mse},
        },
        "history": history_dict,
        "paths": {
            "best_keras": str(model_dir / "model.keras"),
            "final_keras": str(final_keras),
            "savedmodel": str(savedmodel_dir),
        },
    }
    write_json(out_dir / "metrics.json", metrics)

    print(f"[DONE] metrics -> {out_dir / 'metrics.json'}")
    print(f"[DONE] model   -> {model_dir}")


if __name__ == "__main__":
    main()
