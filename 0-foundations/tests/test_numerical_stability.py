from __future__ import annotations

import numpy as np
import pytest

from src.numerical_stability import (
    clip_gradients_numpy,
    clip_gradients_torch,
    naive_log_sum_exp,
    stable_log_sum_exp,
    stable_softmax,
)


def test_log_sum_exp_stable_is_finite_for_large_logits() -> None:
    logits = np.array([1000.0, 1001.0, 1002.0], dtype=np.float64)

    with np.errstate(over="ignore", invalid="ignore"):
        naive = naive_log_sum_exp(logits)

    stable = stable_log_sum_exp(logits)

    assert not np.isfinite(naive)
    assert np.isfinite(stable)


def test_stable_softmax_is_valid_distribution() -> None:
    logits = np.array([1000.0, 1001.0, 1002.0], dtype=np.float64)
    probs = stable_softmax(logits)

    assert np.all(probs >= 0.0)
    assert np.isclose(float(np.sum(probs)), 1.0)


def test_numpy_gradient_clipping_bounds_norm() -> None:
    grads = np.array([3000.0, 4000.0], dtype=np.float64)
    clipped, before, after = clip_gradients_numpy(grads, max_norm=10.0)

    assert before > 10.0
    assert after <= 10.0 + 1e-6
    assert clipped.shape == grads.shape


def test_torch_gradient_clipping_bounds_norm() -> None:
    torch = pytest.importorskip("torch")
    nn = pytest.importorskip("torch.nn")

    layer = nn.Linear(3, 2)
    for parameter in layer.parameters():
        parameter.grad = torch.full_like(parameter.data, 1000.0)

    before, after = clip_gradients_torch(layer.parameters(), max_norm=5.0)

    assert before > 5.0
    assert after <= 5.0 + 1e-5
