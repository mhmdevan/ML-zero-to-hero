"""
Numerical stability demonstrations and utilities.

Topics covered:
- Overflow-safe log-sum-exp
- Overflow-safe softmax
- Gradient clipping (NumPy and PyTorch)

Run:
    python -m src.numerical_stability
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import numpy as np

if TYPE_CHECKING:
    from torch import nn


def naive_log_sum_exp(logits: np.ndarray) -> float:
    """Compute log(sum(exp(logits))) naively (can overflow)."""
    values = np.asarray(logits, dtype=np.float64)
    return float(np.log(np.sum(np.exp(values))))


def stable_log_sum_exp(logits: np.ndarray) -> float:
    """Compute log(sum(exp(logits))) with the log-sum-exp trick."""
    values = np.asarray(logits, dtype=np.float64)
    max_value = np.max(values)
    shifted = values - max_value
    return float(max_value + np.log(np.sum(np.exp(shifted))))


def naive_softmax(logits: np.ndarray) -> np.ndarray:
    """Compute softmax naively (can overflow)."""
    values = np.asarray(logits, dtype=np.float64)
    exp_values = np.exp(values)
    return exp_values / np.sum(exp_values)


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    """Compute softmax stably by shifting by max(logits)."""
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values)


def clip_gradients_numpy(
    gradients: np.ndarray,
    max_norm: float,
    eps: float = 1e-12,
) -> tuple[np.ndarray, float, float]:
    """
    Clip a NumPy gradient vector by global L2 norm.

    Returns:
        clipped_gradients, original_norm, clipped_norm
    """
    if max_norm <= 0:
        raise ValueError("max_norm must be positive.")

    grad = np.asarray(gradients, dtype=np.float64)
    original_norm = float(np.linalg.norm(grad, ord=2))

    if original_norm <= max_norm or original_norm < eps:
        return grad.copy(), original_norm, original_norm

    scale = max_norm / (original_norm + eps)
    clipped = grad * scale
    clipped_norm = float(np.linalg.norm(clipped, ord=2))
    return clipped, original_norm, clipped_norm


def torch_grad_norm(parameters: Iterable["nn.Parameter"]) -> float:
    """Compute global L2 norm of gradients for a list of parameters."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for torch_grad_norm.") from exc

    squared = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        grad_norm = torch.linalg.vector_norm(param.grad.detach(), ord=2).item()
        squared += grad_norm**2
    return float(np.sqrt(squared))


def clip_gradients_torch(
    parameters: Iterable["nn.Parameter"],
    max_norm: float,
) -> tuple[float, float]:
    """
    Clip PyTorch gradients by global L2 norm.

    Returns:
        original_norm, clipped_norm
    """
    if max_norm <= 0:
        raise ValueError("max_norm must be positive.")

    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for clip_gradients_torch.") from exc

    params = list(parameters)
    original_norm = torch_grad_norm(params)
    torch.nn.utils.clip_grad_norm_(params, max_norm=max_norm)
    clipped_norm = torch_grad_norm(params)
    return original_norm, clipped_norm


def demo_log_sum_exp() -> None:
    logits = np.array([1000.0, 1001.0, 1002.0], dtype=np.float64)

    with np.errstate(over="ignore", invalid="ignore"):
        naive_value = naive_log_sum_exp(logits)
    stable_value = stable_log_sum_exp(logits)

    print("[log-sum-exp demo]")
    print(f"  logits: {logits.tolist()}")
    print(f"  naive_log_sum_exp:  {naive_value}")
    print(f"  stable_log_sum_exp: {stable_value:.6f}")


def demo_softmax_overflow() -> None:
    logits = np.array([1000.0, 1001.0, 1002.0], dtype=np.float64)

    with np.errstate(over="ignore", invalid="ignore"):
        naive_probs = naive_softmax(logits)
    stable_probs = stable_softmax(logits)

    print("[softmax demo]")
    print(f"  naive_softmax:  {naive_probs}")
    print(f"  stable_softmax: {stable_probs}")
    print(f"  stable_softmax sum: {float(np.sum(stable_probs)):.6f}")


def demo_gradient_clipping() -> None:
    grad = np.array([3000.0, 4000.0], dtype=np.float64)
    clipped, original_norm, clipped_norm = clip_gradients_numpy(grad, max_norm=10.0)

    print("[numpy grad clipping demo]")
    print(f"  original_norm: {original_norm:.6f}")
    print(f"  clipped_norm:  {clipped_norm:.6f}")
    print(f"  clipped_grad:  {clipped}")

    try:
        import torch
        from torch import nn
    except ImportError:
        print("[torch grad clipping demo]")
        print("  skipped: PyTorch is not installed in this environment.")
        return

    linear = nn.Linear(4, 1, bias=False)
    for param in linear.parameters():
        param.grad = torch.full_like(param.data, 5000.0)

    before, after = clip_gradients_torch(linear.parameters(), max_norm=5.0)
    print("[torch grad clipping demo]")
    print(f"  original_norm: {before:.6f}")
    print(f"  clipped_norm:  {after:.6f}")


def main() -> None:
    print("=== Numerical Stability ===")
    demo_log_sum_exp()
    print()
    demo_softmax_overflow()
    print()
    demo_gradient_clipping()


if __name__ == "__main__":
    main()
