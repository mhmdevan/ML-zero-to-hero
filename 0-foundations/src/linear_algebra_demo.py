#!/usr/bin/env python

"""
linear_algebra_demo.py

Goal:
    - Solve a linear system Ax = b
    - Compute determinant and inverse of a small matrix
    - Compute eigenvalues and eigenvectors (basic intuition)

Run:
    python src/linear_algebra_demo.py
"""

import numpy as np
from pathlib import Path


def print_section(title: str):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main():
    # --------------------------------
    # 1) Define a 2x2 matrix A and a vector b
    # --------------------------------
    print_section("1) Linear system Ax = b")

    A = np.array([
        [3.0, 1.0],
        [1.0, 2.0],
    ])
    b = np.array([9.0, 8.0])

    print("Matrix A:\n", A)
    print("Vector b:", b)

    # Solve Ax = b
    x = np.linalg.solve(A, b)
    print("Solution x (such that A @ x = b):", x)

    # Check: A @ x should be equal (or very close) to b
    b_hat = A @ x
    print("A @ x =", b_hat)

    # --------------------------------
    # 2) Determinant and inverse
    # --------------------------------
    print_section("2) Determinant and inverse")

    det_A = np.linalg.det(A)
    print("det(A) =", det_A)

    if abs(det_A) < 1e-8:
        print("Matrix A is (almost) singular → no inverse.")
    else:
        A_inv = np.linalg.inv(A)
        print("A inverse:\n", A_inv)
        print("A @ A_inv:\n", A @ A_inv)  # should be close to identity

    # --------------------------------
    # 3) Eigenvalues and eigenvectors
    # --------------------------------
    print_section("3) Eigenvalues and eigenvectors")

    # We use the same A for eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eig(A)

    print("Eigenvalues:", eigenvalues)
    print("Eigenvectors (columns):\n", eigenvectors)

    # Quick intuition check for the first eigenpair:
    # A @ v ≈ λ * v
    lambda_0 = eigenvalues[0]
    v0 = eigenvectors[:, 0]
    left = A @ v0
    right = lambda_0 * v0

    print("\nCheck for the first eigenpair (lambda_0, v0):")
    print("A @ v0 =", left)
    print("lambda_0 * v0 =", right)


if __name__ == "__main__":
    main()
