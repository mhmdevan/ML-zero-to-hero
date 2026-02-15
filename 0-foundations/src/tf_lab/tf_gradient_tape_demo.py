#!/usr/bin/env python
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from .tf_utils import set_env_quiet_tf, seed_everything


@dataclass
class GDConfig:
    n_samples: int = 200
    true_w: float = 3.0
    true_b: float = 2.0
    noise_std: float = 0.7
    lr: float = 0.05
    n_steps: int = 200
    seed: int = 42


def make_synthetic(cfg: GDConfig) -> tuple[tf.Tensor, tf.Tensor]:
    tf.random.set_seed(cfg.seed)
    x = tf.linspace(-2.0, 2.0, cfg.n_samples)
    x = tf.reshape(x, (-1, 1))
    noise = tf.random.normal(tf.shape(x), stddev=cfg.noise_std)
    y = cfg.true_w * x + cfg.true_b + noise
    return x, y


def run_manual_gd(cfg: GDConfig) -> None:
    print("\n=== TF Manual GD with GradientTape ===")
    x, y = make_synthetic(cfg)

    w = tf.Variable(tf.random.normal(shape=(1, 1), seed=cfg.seed), trainable=True)
    b = tf.Variable(tf.zeros(shape=(1,)), trainable=True)

    for step in range(1, cfg.n_steps + 1):
        with tf.GradientTape() as tape:
            y_pred = x * w + b
            loss = tf.reduce_mean(tf.square(y_pred - y))

        dw, db = tape.gradient(loss, [w, b])
        w.assign_sub(cfg.lr * dw)
        b.assign_sub(cfg.lr * db)

        if step == 1 or step % 20 == 0:
            print(
                f"[manual] step={step:03d} "
                f"loss={loss.numpy():.4f} "
                f"w={w.numpy().item():.3f} b={b.numpy().item():.3f}"
            )

    print(f"[manual] final w={w.numpy().item():.3f} b={b.numpy().item():.3f} (true {cfg.true_w},{cfg.true_b})")


def run_keras_fit(cfg: GDConfig) -> None:
    print("\n=== TF Keras model.fit ===")
    x, y = make_synthetic(cfg)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(1,)),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=cfg.lr),
        loss="mse",
        metrics=[tf.keras.metrics.MeanSquaredError(name="mse")],
    )

    hist = model.fit(x, y, epochs=10, batch_size=32, verbose=0)
    w, b = model.layers[-1].get_weights()
    print(f"[keras] final loss={hist.history['loss'][-1]:.4f} w={float(w.squeeze()):.3f} b={float(b.squeeze()):.3f}")


@tf.function
def tf_graph_predict(model: tf.keras.Model, x: tf.Tensor) -> tf.Tensor:
    return model(x, training=False)


def benchmark_eager_vs_graph() -> None:
    print("\n=== Eager vs @tf.function latency ===")
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(64,)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    x = tf.random.normal((4096, 64))

    # warmup
    _ = model(x)
    _ = tf_graph_predict(model, x)

    def timeit(fn, runs: int = 50) -> tuple[float, float]:
        ts = []
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = fn()
            t1 = time.perf_counter()
            ts.append(t1 - t0)
        return float(np.mean(ts) * 1000), float(np.quantile(ts, 0.95) * 1000)

    eager_avg, eager_p95 = timeit(lambda: model(x), runs=50)
    graph_avg, graph_p95 = timeit(lambda: tf_graph_predict(model, x), runs=50)

    print(f"eager: avg={eager_avg:.3f}ms p95={eager_p95:.3f}ms")
    print(f"graph: avg={graph_avg:.3f}ms p95={graph_p95:.3f}ms")


def main() -> None:
    set_env_quiet_tf()
    seed_everything(42)

    cfg = GDConfig()
    run_manual_gd(cfg)
    run_keras_fit(cfg)
    benchmark_eager_vs_graph()


if __name__ == "__main__":
    main()
