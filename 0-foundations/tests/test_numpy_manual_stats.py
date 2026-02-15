from __future__ import annotations

import numpy as np
import pytest

from src.numpy_manual_stats import (
    manual_mean,
    manual_std,
    manual_variance,
    standardize_1d,
    standardize_features,
)


def test_manual_mean_matches_numpy() -> None:
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    assert np.isclose(float(manual_mean(x)), np.mean(x))
    assert np.allclose(np.asarray(manual_mean(x, axis=0)), np.mean(x, axis=0))
    assert np.allclose(np.asarray(manual_mean(x, axis=1)), np.mean(x, axis=1))


def test_manual_variance_and_std_match_numpy() -> None:
    rng = np.random.default_rng(seed=7)
    x = rng.normal(size=(20, 4))

    assert np.allclose(np.asarray(manual_variance(x, axis=0)), np.var(x, axis=0, ddof=0))
    assert np.allclose(np.asarray(manual_std(x, axis=1)), np.std(x, axis=1, ddof=0))


def test_standardize_features_center_and_scale_columns() -> None:
    rng = np.random.default_rng(seed=42)
    x = rng.normal(loc=[100.0, 10.0], scale=[15.0, 3.0], size=(200, 2))

    z, means, stds = standardize_features(x)

    assert means.shape == (2,)
    assert stds.shape == (2,)
    assert np.allclose(np.mean(z, axis=0), np.zeros(2), atol=1e-7)
    assert np.allclose(np.std(z, axis=0), np.ones(2), atol=1e-7)


def test_standardize_1d_rejects_non_1d_input() -> None:
    with pytest.raises(ValueError):
        standardize_1d(np.array([[1.0, 2.0], [3.0, 4.0]]))
