"""
torch_lstm_timeseries.py

Univariate time-series forecasting with LSTM vs a naïve baseline
on a synthetic "sales" series.

Goal:
    - Show a clean, production-like time-series pipeline:
        * generate synthetic but realistic sales series
        * time-based train/test split
        * StandardScaler fit only on train
        * sliding windows (input_window -> horizon=1)
        * LSTM model vs naïve "last value" baseline
        * MSE comparison + CSV + plot

Usage:
    python -m src.torch_lstm_timeseries
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------


@dataclass
class DataConfig:
    series_length: int = 360  # number of time steps
    train_ratio: float = 0.8  # 80% train, 20% test
    input_window: int = 30    # how many past steps to look at
    horizon: int = 1          # forecast 1 step ahead
    seed: int = 42


@dataclass
class ModelConfig:
    input_size: int = 1
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.1


@dataclass
class TrainConfig:
    batch_size: int = 32
    num_epochs: int = 120
    lr: float = 1e-3


# ---------------------------------------------------------
# 2. Synthetic "sales" time series
# ---------------------------------------------------------


def generate_synthetic_sales_series(
    length: int,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate a synthetic daily sales series with:
      - slow upward trend,
      - weekly seasonality (weekdays vs weekend),
      - monthly promotions (small bursts every 30 days),
      - Gaussian noise.

    This is intentionally more complex than a pure sine wave so that:
      - naïve last-value baseline is still decent,
      - but a sequence model *can* in principle learn the pattern
        better than just copying the last point.
    """
    rng = random.Random(seed)
    values = []

    base_level = 100.0

    for t in range(length):
        # slow upward trend
        trend = 0.05 * t

        # weekly pattern (period ~7)
        weekly = 10.0 * math.sin(2.0 * math.pi * (t % 7) / 7.0)

        # monthly-ish pattern (period ~30)
        monthly = 5.0 * math.sin(2.0 * math.pi * (t % 30) / 30.0)

        # promotions: first 3 days of each 30-day block
        promo = 0.0
        if t % 30 in (0, 1, 2):
            promo += 25.0

        # noise (non-trivial, so baseline is not "perfect")
        noise = rng.gauss(0.0, 8.0)

        value = base_level + trend + weekly + monthly + promo + noise
        values.append(value)

    series = np.array(values, dtype=np.float32)
    return series


# ---------------------------------------------------------
# 3. Sliding windows (sequence -> target)
# ---------------------------------------------------------


def create_sliding_windows(
    series_scaled: np.ndarray,
    input_window: int,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build sliding windows over a *scaled* series.

    Args:
        series_scaled: 1D numpy array (already scaled).
        input_window: number of past points to use as input.
        horizon: forecast horizon (here 1).

    Returns:
        X: shape (num_windows, input_window)
        y: shape (num_windows, 1)
        target_indices: shape (num_windows,), index in the original series
                        corresponding to the target y.
    """
    assert series_scaled.ndim == 1, "series_scaled must be 1D."

    X_list = []
    y_list = []
    idx_list = []

    n = len(series_scaled)
    last_start = n - input_window - horizon + 1
    if last_start <= 0:
        raise ValueError(
            f"Series too short for given input_window={input_window}, horizon={horizon}"
        )

    for start in range(last_start):
        end = start + input_window
        target_idx = end + horizon - 1

        window = series_scaled[start:end]
        target_value = series_scaled[target_idx]

        X_list.append(window)
        y_list.append(target_value)
        idx_list.append(target_idx)

    X = np.stack(X_list, axis=0)  # (num_windows, input_window)
    y = np.array(y_list, dtype=np.float32).reshape(-1, 1)
    idx = np.array(idx_list, dtype=np.int64)

    return X, y, idx


# ---------------------------------------------------------
# 4. LSTM model definition
# ---------------------------------------------------------


class SalesLSTM(nn.Module):
    """
    Simple univariate LSTM forecaster:
    input:  (batch, seq_len, 1)
    output: (batch, 1)  -> next value
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=cfg.input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Linear(cfg.hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        out, (h_n, c_n) = self.lstm(x)
        # We only need the last time step's hidden state
        last_hidden = out[:, -1, :]  # (batch, hidden_size)
        y_pred = self.fc(last_hidden)  # (batch, 1)
        return y_pred


# ---------------------------------------------------------
# 5. Training & evaluation
# ---------------------------------------------------------


def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    device: torch.device,
) -> SalesLSTM:
    """
    Train the LSTM on scaled sliding windows.
    """
    model = SalesLSTM(model_cfg).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr)

    # build DataLoader
    X_train_t = torch.from_numpy(X_train).float().unsqueeze(-1)  # (N, seq_len, 1)
    y_train_t = torch.from_numpy(y_train).float()  # (N, 1)

    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=train_cfg.batch_size, shuffle=True)

    num_epochs = train_cfg.num_epochs
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        num_samples = 0

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            y_pred = model(xb)
            loss = criterion(y_pred, yb)
            loss.backward()
            optimizer.step()

            batch_size = xb.size(0)
            epoch_loss += loss.item() * batch_size
            num_samples += batch_size

        epoch_loss /= max(num_samples, 1)

        if epoch == 1 or epoch % 10 == 0:
            print(f"[TRAIN] epoch={epoch:03d} | train_loss={epoch_loss:.4f}")

    return model


@torch.no_grad()
def evaluate_lstm_vs_baseline(
    series: np.ndarray,
    scaler: StandardScaler,
    X_test: np.ndarray,
    y_test: np.ndarray,
    target_indices_test: np.ndarray,
    model: SalesLSTM,
    device: torch.device,
    project_root: Path,
) -> Dict[str, float]:
    """
    Evaluate the trained LSTM vs a naïve last-value baseline on the test windows.

    Args:
        series: original unscaled series (1D).
        scaler: fitted StandardScaler.
        X_test: scaled inputs (num_test, input_window).
        y_test: scaled targets (num_test, 1).
        target_indices_test: indices of targets in the original series.
        model: trained LSTM.
        device: torch device.
        project_root: repo root for saving outputs.

    Returns:
        dict with 'lstm_mse' and 'baseline_mse'.
    """
    model.eval()

    # Convert test windows to tensor
    X_test_t = torch.from_numpy(X_test).float().unsqueeze(-1).to(device)  # (N_test, seq_len, 1)

    # LSTM predictions (scaled)
    y_pred_scaled_t = model(X_test_t)  # (N_test, 1)
    y_pred_scaled = y_pred_scaled_t.cpu().numpy()

    # Inverse scale
    y_true_scaled = y_test  # (N_test, 1)
    y_true_unscaled = scaler.inverse_transform(y_true_scaled)
    y_pred_unscaled = scaler.inverse_transform(y_pred_scaled)

    # Baseline: last value of each window, unscaled
    last_values_scaled = X_test[:, -1].reshape(-1, 1)
    last_values_unscaled = scaler.inverse_transform(last_values_scaled)

    lstm_mse = mean_squared_error(y_true_unscaled, y_pred_unscaled)
    baseline_mse = mean_squared_error(y_true_unscaled, last_values_unscaled)

    print(f"[TEST] LSTM MSE={lstm_mse:.4f}")
    print(f"[BASELINE] Naive last-value MSE={baseline_mse:.4f}")
    print(
        f"[RESULT] LSTM MSE={lstm_mse:.4f} vs Naive baseline MSE={baseline_mse:.4f} "
        f"(diff={lstm_mse - baseline_mse:.4f})"
    )

    # -------------------------------------------------
    # Save CSV with predictions vs true
    # -------------------------------------------------
    output_dir = project_root / "output" / "timeseries"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "lstm_predictions_vs_true.csv"

    df_pred = pd.DataFrame(
        {
            "target_index": target_indices_test,
            "true_value": y_true_unscaled.flatten(),
            "lstm_pred": y_pred_unscaled.flatten(),
            "baseline_last_value": last_values_unscaled.flatten(),
        }
    )
    df_pred.to_csv(csv_path, index=False)
    print(f"[SAVE] Saved LSTM predictions vs true series to {csv_path}")

    # -------------------------------------------------
    # Plot full series + test predictions
    # -------------------------------------------------
    plot_path = output_dir / "lstm_predictions_plot.png"

    plt.figure(figsize=(12, 6))
    plt.plot(range(len(series)), series, label="True series", linewidth=1.0)

    # Only mark test targets to not clutter the plot
    plt.scatter(
        target_indices_test,
        y_true_unscaled.flatten(),
        label="True (test targets)",
        s=20,
        alpha=0.7,
    )
    plt.scatter(
        target_indices_test,
        y_pred_unscaled.flatten(),
        label="LSTM predictions",
        s=20,
        alpha=0.7,
    )
    plt.scatter(
        target_indices_test,
        last_values_unscaled.flatten(),
        label="Naive last-value",
        s=20,
        alpha=0.7,
    )

    plt.title("Synthetic sales series: true vs LSTM vs naive baseline (test region)")
    plt.xlabel("Time index")
    plt.ylabel("Sales")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=140)
    plt.close()

    print(f"[PLOT] Saved series vs LSTM predictions to {plot_path}")

    return {
        "lstm_mse": float(lstm_mse),
        "baseline_mse": float(baseline_mse),
    }


# ---------------------------------------------------------
# 6. Main pipeline
# ---------------------------------------------------------


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent

    data_cfg = DataConfig()
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()

    # Reproducibility
    random.seed(data_cfg.seed)
    np.random.seed(data_cfg.seed)
    torch.manual_seed(data_cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # 1) Generate synthetic series
    series = generate_synthetic_sales_series(
        length=data_cfg.series_length,
        seed=data_cfg.seed,
    )
    print(
        f"[DATA] Generated synthetic sales series of length {len(series)} "
        f"(min={series.min():.2f}, max={series.max():.2f})"
    )

    # 2) Time-based train/test split (no shuffling)
    n = len(series)
    split_idx = int(n * data_cfg.train_ratio)
    train_series = series[:split_idx]
    test_series = series[split_idx:]
    print(
        f"[DATA] Train series length={len(train_series)}, "
        f"Test series length={len(test_series)}"
    )

    # 3) Scaling: fit StandardScaler ONLY on train
    scaler = StandardScaler()
    scaler.fit(train_series.reshape(-1, 1))
    series_scaled = scaler.transform(series.reshape(-1, 1)).flatten()

    train_mean = float(scaler.mean_[0])
    train_std = float(scaler.scale_[0])
    print(
        f"[SCALE] Fitted StandardScaler on train series (mean={train_mean:.3f}, std={train_std:.3f})"
    )

    # 4) Sliding windows over the full scaled series
    X_all, y_all, target_indices_all = create_sliding_windows(
        series_scaled,
        input_window=data_cfg.input_window,
        horizon=data_cfg.horizon,
    )
    num_windows = X_all.shape[0]
    print(
        f"[DATA] Sliding windows: X.shape={X_all.shape}, y.shape={y_all.shape}, "
        f"input_window={data_cfg.input_window}, horizon={data_cfg.horizon}"
    )

    # 5) Time-based split on windows:
    #    train windows = those whose target index < split_idx
    #    test windows = those whose target index >= split_idx
    train_mask = target_indices_all < split_idx
    test_mask = ~train_mask

    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_test, y_test = X_all[test_mask], y_all[test_mask]
    target_indices_test = target_indices_all[test_mask]

    print(
        f"[DATA] train_samples={X_train.shape[0]}, "
        f"test_samples={X_test.shape[0]}, "
        f"input_window={data_cfg.input_window}"
    )

    # 6) Train LSTM on train windows
    model = train_lstm(
        X_train=X_train,
        y_train=y_train,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        device=device,
    )

    # 7) Evaluate vs naïve baseline and save outputs
    _metrics = evaluate_lstm_vs_baseline(
        series=series,
        scaler=scaler,
        X_test=X_test,
        y_test=y_test,
        target_indices_test=target_indices_test,
        model=model,
        device=device,
        project_root=project_root,
    )

    print(
        f"[SUMMARY] LSTM MSE={_metrics['lstm_mse']:.4f}, "
        f"Naive MSE={_metrics['baseline_mse']:.4f}"
    )


if __name__ == "__main__":
    main()
