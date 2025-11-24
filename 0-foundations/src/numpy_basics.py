#!/usr/bin/env python

"""
numpy_basics.py

Goal:
    - Introduce numpy arrays (ndarray)
    - Work with vectors and matrices
    - Compute dot products and matrix multiplication
    - See transpose and broadcasting in action

Run:
    python src/numpy_basics.py
"""

import numpy as np
from pathlib import Path


def print_section(title: str):
    """Helper function to print a readable section header."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def save_array_as_text(array: np.ndarray, file_path: Path, header: str = ""):
    """
    Save a numpy array to a .txt file for inspection.

    Parameters
    ----------
    array : np.ndarray
        Array to save.
    file_path : Path
        Where to save the file.
    header : str
        Optional header text at the top of the file.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(file_path, array, fmt="%.4f", header=header)
    print(f"[FILE] Saved array to {file_path}")


def main():
    # --------------------------------
    # 1) Creating simple vectors
    # --------------------------------
    print_section("1) Basic vectors")

    # A 1D vector with 3 elements
    v = np.array([1.0, 2.0, 3.0])
    print("Vector v:", v)
    print("Shape of v:", v.shape)  # (3,)

    # Another vector
    w = np.array([4.0, 5.0, 6.0])
    print("Vector w:", w)
    print("Shape of w:", w.shape)

    # Manual dot product (just to understand)
    # v · w = 1*4 + 2*5 + 3*6
    manual_dot = v[0] * w[0] + v[1] * w[1] + v[2] * w[2]
    print("Manual dot product v · w:", manual_dot)

    # Numpy dot product
    numpy_dot = np.dot(v, w)
    print("Numpy dot product v · w:", numpy_dot)

    # --------------------------------
    # 2) Matrices and matrix multiplication
    # --------------------------------
    print_section("2) Matrices and matrix multiplication")

    # A 2x3 matrix (2 rows, 3 columns)
    A = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ])
    print("Matrix A:\n", A)
    print("Shape of A:", A.shape)  # (2, 3)

    # A 3x2 matrix (3 rows, 2 columns)
    B = np.array([
        [1.0, 2.0],
        [0.0, 1.0],
        [1.0, 0.0],
    ])
    print("Matrix B:\n", B)
    print("Shape of B:", B.shape)  # (3, 2)

    # Matrix multiplication C = A @ B
    # A is (2, 3), B is (3, 2) → C is (2, 2)
    C = A @ B
    print("Matrix C = A @ B:\n", C)
    print("Shape of C:", C.shape)

    # --------------------------------
    # 3) Transpose
    # --------------------------------
    print_section("3) Transpose")

    # Transpose of A: swap rows and columns
    A_T = A.T
    print("A.T (transpose of A):\n", A_T)
    print("Shape of A_T:", A_T.shape)  # (3, 2)

    # --------------------------------
    # 4) Broadcasting
    # --------------------------------
    print_section("4) Broadcasting")

    # Broadcasting means numpy can automatically stretch smaller arrays
    # when adding / multiplying with larger arrays, if shapes are compatible.
    row_vector = np.array([1.0, 10.0, 100.0])  # shape (3,)
    print("Row vector:", row_vector)
    print("Shape of row vector:", row_vector.shape)

    # Add this row vector to each row of A
    # A.shape = (2, 3), row_vector.shape = (3,)
    # Numpy will 'broadcast' row_vector to shape (2, 3) internally.
    A_plus_row = A + row_vector
    print("A + row_vector:\n", A_plus_row)

    # --------------------------------
    # 5) Save some arrays to text files for GitHub
    # --------------------------------
    print_section("5) Saving arrays to text files")

    plots_dir = Path(__file__).resolve().parent.parent / "plots"
    save_array_as_text(A, plots_dir / "numpy_A_matrix.txt", header="Matrix A")
    save_array_as_text(B, plots_dir / "numpy_B_matrix.txt", header="Matrix B")
    save_array_as_text(C, plots_dir / "numpy_C_matrix.txt", header="Matrix C = A @ B")


if __name__ == "__main__":
    main()
