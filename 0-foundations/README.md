# 🧠 ML Project 0 – Python & Math Foundations

Hands-on foundations for Machine Learning in pure Python scripts (no notebooks).  
The goal of this project is to **really understand** the building blocks of ML:

- Python for ML:
  - `numpy` → ndarrays, vector & matrix operations, broadcasting
  - `pandas` → (will be used in later projects, e.g. Project 0.1 – EDA)
  - `matplotlib`, `seaborn` → basic visualizations
- Applied Math:
  - Linear algebra → vectors, matrices, matrix multiplication, transpose, inverse, eigenvalues/eigenvectors (intuitive)
  - Probability & statistics → normal distribution, mean, variance, covariance, correlation
  - Calculus → derivative & gradient, gradient descent in 1D

Each file in `src/` is a small, focused script with clean code and comments in English, designed to be readable on GitHub and to generate visual outputs in the `plots/` folder.

---

## 🗂 Project Structure

```text
ml-project-0-foundations/
    ├─ src/
    │   ├─ numpy_basics.py               # vectors, matrices, dot, matmul, transpose, broadcasting
    │   ├─ linear_algebra_demo.py        # solving Ax=b, determinant, inverse, eigenvalues/vectors
    │   ├─ stats_basics.py               # mean, variance, covariance, correlation, normal dist + plots
    │   └─ gradient_descent_demo.py      # 1D gradient descent on f(w) = (w - 3)^2
    ├─ plots/
    │   ├─ numpy_A_matrix.txt
    │   ├─ numpy_B_matrix.txt
    │   ├─ numpy_C_matrix.txt
    │   ├─ hist_heights.png
    │   ├─ hist_weights.png
    │   ├─ scatter_height_vs_weight.png
    │   └─ gradient_descent_1d.png
    ├─ requirements.txt
    └─ README.md
```

---

## ⚙️ Setup

```bash

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

---

## ▶️ Scripts

### 1. `numpy_basics.py`

**Topics:**

- `ndarray` and `.shape`
- dot product vs manual dot
- matrix multiplication (`@`)
- transpose (`.T`)
- broadcasting (adding a row vector to a matrix)

**Run:**

```bash
python src/numpy_basics.py
```

**Outputs:**

- Prints vector and matrix shapes and dot products
- Saves:
  - `plots/numpy_A_matrix.txt`
  - `plots/numpy_B_matrix.txt`
  - `plots/numpy_C_matrix.txt`

---

### 2. `linear_algebra_demo.py`

**Topics:**

- Solving linear systems \(Ax = b\)
- Determinant and inverse of a 2×2 matrix
- Eigenvalues & eigenvectors, and the relation \(A v = \lambda v\)

**Run:**

```bash
python src/linear_algebra_demo.py
```

**Outputs:**

- Printed solution `x` such that `A @ x ≈ b`
- `det(A)`, `A^{-1}`, and `A @ A^{-1}` (close to identity)
- Eigenvalues and eigenvectors of `A`

---

### 3. `stats_basics.py`

**Topics:**

- Generating random data from a normal distribution
- Mean, variance, standard deviation
- Covariance and correlation between height and weight
- Basic histograms and scatter plot

**Run:**

```bash
python src/stats_basics.py
```

**Visuals (auto-saved):**

#### Histogram of Heights

![Histogram of Heights](plots/hist_heights.png)

#### Histogram of Weights

![Histogram of Weights](plots/hist_weights.png)

#### Height vs Weight Scatter

![Height vs Weight](plots/scatter_height_vs_weight.png)

---

### 4. `gradient_descent_demo.py`

**Topics:**

- 1D objective function: \(f(w) = (w - 3)^2\)
- Analytic derivative: \(f'(w) = 2(w - 3)\)
- Gradient descent update:
  \[
  w_{\text{new}} = w_{\text{old}} - \eta \cdot f'(w_{\text{old}})
  \]

**Run:**

```bash
python src/gradient_descent_demo.py
```

**Visual:**

Gradient descent steps (red dots) moving towards the minimum at \(w = 3\):

![Gradient Descent 1D](plots/gradient_descent_1d.png)

---

## 🧮 Applied Math Summary (Intuition)

- **Vector** → an ordered list of numbers (direction + magnitude).
- **Matrix** → a grid of numbers; can represent a linear transformation.
- **Matrix multiplication** → applying one transformation after another.
- **Determinant** → how much the matrix scales area/volume; zero means it collapses space (no inverse).
- **Eigenvector** → a direction that the matrix only stretches, not rotates.  
  **Eigenvalue** → how much it stretches that direction.

- **Mean** → average.
- **Variance** → how wide the data is spread around the mean.
- **Std (standard deviation)** → square root of variance; spread in original units.
- **Covariance** → do two variables move together?
- **Correlation** → normalized covariance, between -1 and 1.

- **Gradient** → vector of partial derivatives; direction of steepest *increase*.
- **Gradient descent** → move in the opposite direction of gradient to *minimize* a function.

---
