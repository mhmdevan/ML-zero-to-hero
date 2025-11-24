#!/usr/bin/env python

"""
gradient_descent_demo.py

Goal:
    - Show how gradient descent works on a simple 1D function
    - Function: f(w) = (w - 3)^2
    - Analytic gradient: f'(w) = 2 * (w - 3)

Run:
    python src/gradient_descent_demo.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def ensure_plots_dir() -> Path:
    plots_dir = Path(__file__).resolve().parent.parent / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir


def print_section(title: str):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def f(w: float) -> float:
    """Objective function: f(w) = (w - 3)^2."""
    return (w - 3.0) ** 2


def grad_f(w: float) -> float:
    """Analytic gradient (derivative) of f: f'(w) = 2 * (w - 3)."""
    return 2.0 * (w - 3.0)


def run_gradient_descent(
    w_init: float,
    learning_rate: float,
    n_steps: int,
):
    """
    Run gradient descent on f(w) starting from w_init.

    Parameters
    ----------
    w_init : float
        Initial value of the parameter w.
    learning_rate : float
        Step size (eta). Controls how big each update is.
    n_steps : int
        Number of gradient descent iterations.

    Returns
    -------
    history_w : list of float
        Values of w at each step.
    history_f : list of float
        Values of f(w) at each step.
    """
    w = w_init
    history_w = [w]
    history_f = [f(w)]

    for step in range(n_steps):
        g = grad_f(w)              # gradient at current w
        w = w - learning_rate * g  # gradient descent update
        history_w.append(w)
        history_f.append(f(w))

    return history_w, history_f


def main():
    plots_dir = ensure_plots_dir()

    print_section("1) Gradient descent on f(w) = (w - 3)^2")

    # Initial parameter
    w_init = -5.0
    learning_rate = 0.1
    n_steps = 30

    print(f"Initial w: {w_init}")
    print(f"Learning rate: {learning_rate}")
    print(f"Number of steps: {n_steps}")

    history_w, history_f = run_gradient_descent(
        w_init=w_init,
        learning_rate=learning_rate,
        n_steps=n_steps,
    )

    for i, (w_val, f_val) in enumerate(zip(history_w, history_f)):
        print(f"Step {i:2d}: w = {w_val:.6f}, f(w) = {f_val:.6f}")

    # --------------------------------
    # 2) Plot the function and the path of gradient descent
    # --------------------------------
    print_section("2) Plotting the function and the descent path")

    w_grid = np.linspace(-6.0, 6.0, 400)
    f_grid = f(w_grid)

    plt.figure(figsize=(8, 5))
    plt.plot(w_grid, f_grid, label="f(w) = (w - 3)^2")
    plt.scatter(history_w, history_f, color="red", s=40, label="GD steps")
    plt.title("Gradient Descent on f(w) = (w - 3)^2")
    plt.xlabel("w")
    plt.ylabel("f(w)")
    plt.legend()
    plt.tight_layout()

    gd_path_path = plots_dir / "gradient_descent_1d.png"
    plt.savefig(gd_path_path, dpi=140)
    plt.close()
    print(f"[PLOT] Saved gradient descent path to {gd_path_path}")


if __name__ == "__main__":
    main()
