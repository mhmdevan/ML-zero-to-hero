"""
D1 – Gradient Descent with PyTorch Autograd

This script re-implements the classic "fit y = 3x + 2" demo using:
  1) Manual gradient descent with autograd
  2) nn.Linear + torch.optim (SGD)

Goal:
  - Show how autograd works
  - Show how optimizers update parameters
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


# -----------------------------
# Global config
# -----------------------------


@dataclass
class GDConfig:
    n_samples: int = 100
    true_w: float = 3.0
    true_b: float = 2.0
    noise_std: float = 0.7
    lr: float = 0.05
    n_steps: int = 200


def make_synthetic_data(cfg: GDConfig, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate synthetic data: y = true_w * x + true_b + noise
    x: shape (n_samples, 1)
    y: shape (n_samples, 1)
    """
    torch.manual_seed(42)

    x = torch.linspace(-2.0, 2.0, steps=cfg.n_samples, device=device).unsqueeze(1)
    noise = torch.randn_like(x) * cfg.noise_std
    y = cfg.true_w * x + cfg.true_b + noise
    return x, y


# -----------------------------
# 1) Manual GD with autograd
# -----------------------------


def run_manual_gd(cfg: GDConfig, device: torch.device = torch.device("cpu")) -> None:
    """
    Manual gradient descent using autograd + no_grad() for parameter updates.
    """
    print("\n=== Manual Gradient Descent with Autograd ===")

    x, y = make_synthetic_data(cfg, device)

    # Initialize parameters as nn.Parameter so autograd tracks them
    w = torch.nn.Parameter(torch.randn(1, device=device))
    b = torch.nn.Parameter(torch.zeros(1, device=device))

    for step in range(1, cfg.n_steps + 1):
        # Forward pass
        y_pred = w * x + b  # broadcasting: (n,1) * (1,) + (1,)
        loss = torch.mean((y_pred - y) ** 2)

        # Reset gradients
        if w.grad is not None:
            w.grad.zero_()
        if b.grad is not None:
            b.grad.zero_()

        # Backprop
        loss.backward()

        # Gradient descent update
        with torch.no_grad():
            w -= cfg.lr * w.grad
            b -= cfg.lr * b.grad

        if step % 20 == 0 or step == 1:
            print(
                f"[manual-gd] step={step:3d} | "
                f"loss={loss.item():.4f} | "
                f"w={w.item():.3f} | b={b.item():.3f}"
            )

    print(
        f"[manual-gd] Final params: w={w.item():.3f}, b={b.item():.3f} "
        f"(true_w={cfg.true_w}, true_b={cfg.true_b})"
    )


# -----------------------------
# 2) nn.Linear + torch.optim
# -----------------------------


def run_nn_linear_gd(
    cfg: GDConfig,
    device: torch.device = torch.device("cpu"),
    optimizer_name: str = "sgd",
) -> None:
    """
    Same problem, but using nn.Linear and a torch.optim optimizer instead of manual updates.
    """
    print(f"\n=== nn.Linear + torch.optim ({optimizer_name.upper()}) ===")

    x, y = make_synthetic_data(cfg, device)

    model = nn.Linear(in_features=1, out_features=1, bias=True).to(device)
    criterion = nn.MSELoss()

    if optimizer_name.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr)
    elif optimizer_name.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    else:
        raise ValueError(f"Unsupported optimizer_name: {optimizer_name}")

    for step in range(1, cfg.n_steps + 1):
        # Forward
        y_pred = model(x)
        loss = criterion(y_pred, y)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 20 == 0 or step == 1:
            w = model.weight.data.item()
            b = model.bias.data.item()
            print(
                f"[{optimizer_name}] step={step:3d} | "
                f"loss={loss.item():.4f} | "
                f"w={w:.3f} | b={b:.3f}"
            )

    w = model.weight.data.item()
    b = model.bias.data.item()
    print(
        f"[{optimizer_name}] Final params: w={w:.3f}, b={b:.3f} "
        f"(true_w={cfg.true_w}, true_b={cfg.true_b})"
    )


# -----------------------------
# Main
# -----------------------------


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    cfg = GDConfig()

    run_manual_gd(cfg, device=device)
    run_nn_linear_gd(cfg, device=device, optimizer_name="sgd")
    run_nn_linear_gd(cfg, device=device, optimizer_name="adam")


if __name__ == "__main__":
    main()
