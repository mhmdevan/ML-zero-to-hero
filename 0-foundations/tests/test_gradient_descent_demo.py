from __future__ import annotations

import numpy as np

from src.gradient_descent_demo import f, grad_f, run_gradient_descent


def test_grad_matches_numeric_derivative() -> None:
    w = 1.7
    eps = 1e-6
    numeric = (float(f(w + eps)) - float(f(w - eps))) / (2 * eps)
    analytic = grad_f(w)
    assert np.isclose(analytic, numeric, atol=1e-5)


def test_gradient_descent_converges_to_minimum() -> None:
    history_w, history_f = run_gradient_descent(w_init=-5.0, learning_rate=0.1, n_steps=30)

    assert len(history_w) == 31
    assert len(history_f) == 31
    assert history_f[-1] < history_f[0]
    assert np.isclose(history_w[-1], 3.0, atol=1e-2)
