#!/usr/bin/env python

"""
stats_basics.py

Goal:
    - Generate random data from a normal distribution
    - Compute mean, variance, standard deviation
    - Compute covariance and correlation between two variables
    - Plot histograms and scatter plot, and save them for GitHub

Run:
    python src/stats_basics.py
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set(style="whitegrid")


def ensure_plots_dir() -> Path:
    """
    Ensure that the 'plots' directory exists and return its Path.
    """
    plots_dir = Path(__file__).resolve().parent.parent / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir


def print_section(title: str):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main():
    plots_dir = ensure_plots_dir()

    # --------------------------------
    # 1) Generate normal data
    # --------------------------------
    print_section("1) Generating normal data")

    np.random.seed(42)  # for reproducibility

    # Example: heights (in cm) following a normal distribution
    # mean ~ 175, std ~ 10
    n_samples = 500
    heights = np.random.normal(loc=175.0, scale=10.0, size=n_samples)

    print("First 10 heights:", heights[:10])
    print("Shape of heights:", heights.shape)

    # --------------------------------
    # 2) Basic statistics: mean, variance, std
    # --------------------------------
    print_section("2) Basic statistics")

    mean_height = np.mean(heights)
    var_height = np.var(heights)       # population variance
    std_height = np.std(heights)       # population std

    print("Mean height:", mean_height)
    print("Variance of height:", var_height)
    print("Std of height:", std_height)

    # --------------------------------
    # 3) Two variables: height and weight
    # --------------------------------
    print_section("3) Two variables: height and weight")

    # Let's create a simple linear relationship with some noise:
    # weight ≈ 0.9 * height - 80 + noise
    noise = np.random.normal(loc=0.0, scale=5.0, size=n_samples)
    weights = 0.9 * heights - 80.0 + noise

    print("First 10 weights:", weights[:10])

    # Covariance matrix: 2x2
    cov_matrix = np.cov(heights, weights, bias=False)
    print("Covariance matrix (heights, weights):\n", cov_matrix)

    # Correlation coefficient (Pearson)
    corr_matrix = np.corrcoef(heights, weights)
    corr_hw = corr_matrix[0, 1]
    print("Correlation between height and weight:", corr_hw)

    # --------------------------------
    # 4) Plots
    # --------------------------------
    print_section("4) Plotting")

    # Histogram of heights
    plt.figure(figsize=(8, 5))
    plt.hist(heights, bins=30, edgecolor="black")
    plt.title("Histogram of Heights")
    plt.xlabel("Height (cm)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    hist_heights_path = plots_dir / "hist_heights.png"
    plt.savefig(hist_heights_path, dpi=140)
    plt.close()
    print(f"[PLOT] Saved heights histogram to {hist_heights_path}")

    # Histogram of weights
    plt.figure(figsize=(8, 5))
    plt.hist(weights, bins=30, edgecolor="black")
    plt.title("Histogram of Weights")
    plt.xlabel("Weight (kg)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    hist_weights_path = plots_dir / "hist_weights.png"
    plt.savefig(hist_weights_path, dpi=140)
    plt.close()
    print(f"[PLOT] Saved weights histogram to {hist_weights_path}")

    # Scatter plot: height vs weight
    plt.figure(figsize=(8, 5))
    plt.scatter(heights, weights, alpha=0.6)
    plt.title("Height vs Weight")
    plt.xlabel("Height (cm)")
    plt.ylabel("Weight (kg)")
    plt.tight_layout()
    scatter_hw_path = plots_dir / "scatter_height_vs_weight.png"
    plt.savefig(scatter_hw_path, dpi=140)
    plt.close()
    print(f"[PLOT] Saved scatter plot to {scatter_hw_path}")


if __name__ == "__main__":
    main()
