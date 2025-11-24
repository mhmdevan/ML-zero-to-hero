#!/usr/bin/env python

"""
numpy_manual_stats.py

Project 0.2 – Manual statistics and standardization with NumPy.

Goal:
    - Implement manual mean / variance / standard deviation
      using basic NumPy operations (no np.mean / np.var / np.std inside).
    - Implement 1D standardization (z-score).
    - Implement feature-wise standardization for 2D data
      shaped as (n_samples, n_features).
    - Build intuition for working with matrices in ML instead of being afraid of them.

Run:
    python src/numpy_manual_stats.py
"""

import numpy as np


# ---------------------------------------------------------------------------
# Helper printing
# ---------------------------------------------------------------------------

def print_section(title: str) -> None:
    """Pretty-print a section header in the terminal."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ---------------------------------------------------------------------------
# Manual statistics implementations
# ---------------------------------------------------------------------------

def manual_mean(x, axis: int | None = None) -> np.ndarray | float:
    """
    Compute the mean of an array using basic NumPy ops.

    Parameters
    ----------
    x : array-like
        Input data.
    axis : int or None
        Axis along which to compute the mean.
        - If None, compute the mean of all elements.
        - If 0, compute the mean column-wise (over rows).
        - If 1, compute the mean row-wise (over columns).

    Returns
    -------
    mean : float or np.ndarray
        - Scalar if axis is None.
        - 1D array if axis is 0 or 1.
    """
    arr = np.asarray(x, dtype=float)

    if axis is None:
        total = np.sum(arr)
        count = arr.size
    else:
        total = np.sum(arr, axis=axis)
        count = arr.shape[axis]

    mean_value = total / count
    return mean_value


def manual_variance(x, axis: int | None = None, ddof: int = 0) -> np.ndarray | float:
    """
    Compute the variance of an array using manual mean and basic NumPy ops.

    Variance formula (population, ddof=0):
        var = (1 / N) * sum( (x_i - mean)^2 )

    If ddof > 0 (like ddof=1), then:
        var = (1 / (N - ddof)) * sum( (x_i - mean)^2 )

    Parameters
    ----------
    x : array-like
        Input data.
    axis : int or None
        Axis along which to compute the variance.
    ddof : int
        Delta degrees of freedom. 0 for population variance,
        1 for sample variance (statistics convention).

    Returns
    -------
    var : float or np.ndarray
        - Scalar if axis is None.
        - 1D array if axis is 0 or 1.
    """
    arr = np.asarray(x, dtype=float)
    mean_value = manual_mean(arr, axis=axis)

    # Broadcasting:
    # - If axis is None, mean_value is scalar → arr - mean_value works on all elements.
    # - If axis=0, mean_value has shape (n_features,) and will be subtracted row-wise.
    # - If axis=1, mean_value has shape (n_samples,) and will be subtracted column-wise.
    diffs = arr - mean_value
    squared = diffs ** 2

    if axis is None:
        sum_squared = np.sum(squared)
        count = arr.size
    else:
        sum_squared = np.sum(squared, axis=axis)
        count = arr.shape[axis]

    denom = count - ddof
    if denom <= 0:
        raise ValueError(
            f"ddof={ddof} is too large for axis with size={count}. "
            "Denominator must be positive."
        )

    var_value = sum_squared / denom
    return var_value


def manual_std(x, axis: int | None = None, ddof: int = 0) -> np.ndarray | float:
    """
    Compute the standard deviation using manual variance.

    std = sqrt(variance)
    """
    var_value = manual_variance(x, axis=axis, ddof=ddof)
    std_value = np.sqrt(var_value)
    return std_value


# ---------------------------------------------------------------------------
# Standardization
# ---------------------------------------------------------------------------

def standardize_1d(x, ddof: int = 0, eps: float = 1e-8):
    """
    Standardize a 1D array: z = (x - mean) / std.

    Parameters
    ----------
    x : array-like
        1D input data.
    ddof : int
        ddof passed to variance/std (0 for population).
    eps : float
        Small value to avoid division by zero if std is extremely small.

    Returns
    -------
    z : np.ndarray
        Standardized data.
    mean_value : float
        Mean used for standardization.
    std_value : float
        Std used for standardization (before epsilon adjustment).
    """
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError("standardize_1d expects a 1D array.")

    mean_value = manual_mean(arr)
    std_value = manual_std(arr, ddof=ddof)

    # Protect against zero (or extremely small) std
    if std_value < eps:
        std_safe = eps
    else:
        std_safe = std_value

    z = (arr - mean_value) / std_safe
    return z, mean_value, std_value


def standardize_features(X, ddof: int = 0, eps: float = 1e-8):
    """
    Standardize a 2D array of shape (n_samples, n_features) along axis=0.

    For each feature/column j:
        mean_j = mean over all samples
        std_j  = std  over all samples
        Z[:, j] = (X[:, j] - mean_j) / std_j

    Parameters
    ----------
    X : array-like
        2D data, shape (n_samples, n_features).
    ddof : int
        ddof for variance/std (0 is common in ML).
    eps : float
        Small value to avoid division by zero when std is extremely small.

    Returns
    -------
    Z : np.ndarray
        Standardized data (same shape as X).
    means : np.ndarray
        Means for each feature (shape: (n_features,)).
    stds : np.ndarray
        Standard deviations for each feature (shape: (n_features,)).
    """
    arr = np.asarray(X, dtype=float)
    if arr.ndim != 2:
        raise ValueError(
            "standardize_features expects a 2D array of shape (n_samples, n_features)."
        )

    # Column-wise mean and std → axis=0
    means = manual_mean(arr, axis=0)
    stds = manual_std(arr, axis=0, ddof=ddof)

    # Avoid division by zero: if a feature has std ≈ 0, we keep it unscaled
    stds_safe = np.where(stds < eps, 1.0, stds)

    Z = (arr - means) / stds_safe
    return Z, means, stds


# ---------------------------------------------------------------------------
# Demo / sanity checks
# ---------------------------------------------------------------------------

def main() -> None:
    print_section("Project 0.2 – Manual NumPy statistics & standardization")

    # --------------------------------------------------
    # 1) Create synthetic 2D data (n_samples x n_features)
    # --------------------------------------------------
    print_section("1) Creating synthetic 2D data (matrix)")

    rng = np.random.default_rng(seed=42)
    n_samples = 6
    n_features = 3

    # Think of columns as: [height, weight, age] just as an example.
    # Different means and stds per feature to force you to think in columns.
    X = rng.normal(
        loc=[170.0, 70.0, 30.0],   # means for each feature
        scale=[10.0, 8.0, 5.0],    # stds for each feature
        size=(n_samples, n_features),
    )

    print("Raw data X (shape: n_samples x n_features):")
    print(X)
    print("Shape of X:", X.shape)

    # --------------------------------------------------
    # 2) Manual vs NumPy stats on a single 1D feature
    # --------------------------------------------------
    print_section("2) Manual vs NumPy stats on a single feature (column 0)")

    feature0 = X[:, 0]  # first column
    print("Feature 0 values:", feature0)

    manual_mean_0 = manual_mean(feature0)
    numpy_mean_0 = np.mean(feature0)

    manual_var_0 = manual_variance(feature0, ddof=0)
    numpy_var_0 = np.var(feature0, ddof=0)

    manual_std_0 = manual_std(feature0, ddof=0)
    numpy_std_0 = np.std(feature0, ddof=0)

    print("\nMeans:")
    print("  manual_mean_0 =", manual_mean_0)
    print("  numpy_mean_0  =", numpy_mean_0)
    print("  difference    =", abs(manual_mean_0 - numpy_mean_0))

    print("\nVariances (ddof=0):")
    print("  manual_var_0  =", manual_var_0)
    print("  numpy_var_0   =", numpy_var_0)
    print("  difference    =", abs(manual_var_0 - numpy_var_0))

    print("\nStandard deviations (ddof=0):")
    print("  manual_std_0  =", manual_std_0)
    print("  numpy_std_0   =", numpy_std_0)
    print("  difference    =", abs(manual_std_0 - numpy_std_0))

    # --------------------------------------------------
    # 3) Column-wise stats with axis=0 (matrix perspective)
    # --------------------------------------------------
    print_section("3) Column-wise mean / var / std (axis=0)")

    manual_means = manual_mean(X, axis=0)
    numpy_means = np.mean(X, axis=0)

    manual_vars = manual_variance(X, axis=0, ddof=0)
    numpy_vars = np.var(X, axis=0, ddof=0)

    manual_stds = manual_std(X, axis=0, ddof=0)
    numpy_stds = np.std(X, axis=0, ddof=0)

    print("Manual means per column:", manual_means)
    print("NumPy  means per column:", numpy_means)
    print("Difference (means):     ", manual_means - numpy_means)

    print("\nManual variances per column:", manual_vars)
    print("NumPy  variances per column:", numpy_vars)
    print("Difference (variances):     ", manual_vars - numpy_vars)

    print("\nManual stds per column:", manual_stds)
    print("NumPy  stds per column:", numpy_stds)
    print("Difference (stds):      ", manual_stds - numpy_stds)

    # --------------------------------------------------
    # 4) Standardize all features (matrix standardization)
    # --------------------------------------------------
    print_section("4) Standardizing all features (z-score per column)")

    Z, means_used, stds_used = standardize_features(X, ddof=0)

    print("Means used for standardization:", means_used)
    print("Stds  used for standardization:", stds_used)
    print("\nStandardized data Z (same shape as X):")
    print(Z)

    # --------------------------------------------------
    # 5) Check that standardized features have ~0 mean and ~1 std
    # --------------------------------------------------
    print_section("5) Checking stats of standardized features")

    Z_means = manual_mean(Z, axis=0)
    Z_stds = manual_std(Z, axis=0, ddof=0)

    print("Mean of each standardized column (should be close to 0):")
    print(Z_means)

    print("\nStd of each standardized column (should be close to 1):")
    print(Z_stds)


if __name__ == "__main__":
    main()
