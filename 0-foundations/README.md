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
- Manual statistics & standardization (Project 0.2):
  - Implement mean / variance / std *by hand* with NumPy
  - Standardization (z-score) for 1D and 2D data
  - Understanding `axis` and broadcasting on matrices

Each file in `src/` is a small, focused script with clean code and comments in English, designed to be readable on GitHub and to generate visual outputs in the `plots/` folder where it makes sense.

---

## 🗂 Project Structure

```text
ml-project-0-foundations/
    ├─ src/
    │   ├─ numpy_basics.py               # vectors, matrices, dot, matmul, transpose, broadcasting
    │   ├─ linear_algebra_demo.py        # solving Ax=b, determinant, inverse, eigenvalues/vectors
    │   ├─ stats_basics.py               # mean, variance, covariance, correlation, normal dist + plots
    │   ├─ gradient_descent_demo.py      # 1D gradient descent on f(w) = (w - 3)^2
    │   └─ numpy_manual_stats.py         # Project 0.2: manual mean/var/std + standardization (1D & 2D)
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
  w_{ ext{new}} = w_{ ext{old}} - \eta \cdot f'(w_{ ext{old}})
  \]

**Run:**

```bash
python src/gradient_descent_demo.py
```

**Visual:**

Gradient descent steps (red dots) moving towards the minimum at \(w = 3\):

![Gradient Descent 1D](plots/gradient_descent_1d.png)

---

### 5. `numpy_manual_stats.py` (Project 0.2 – Manual statistics & standardization)

This script is where you stop being scared of matrices and `axis` arguments.

**Goals:**

- Implement **manual statistics** using NumPy building blocks (no `np.mean`, `np.var`, `np.std` inside):
  - `manual_mean(x, axis=None)`
  - `manual_variance(x, axis=None, ddof=0)`
  - `manual_std(x, axis=None, ddof=0)`
- Implement **standardization (z-score)**:
  - `standardize_1d(x)` for 1D arrays
  - `standardize_features(X)` for 2D data shaped as `(n_samples, n_features)`
- Build real intuition for:
  - how `axis=0` vs `axis=1` work on matrices
  - how broadcasting applies feature-wise means and stds
  - why standardization is used everywhere in ML (e.g. before linear models, logistic regression, neural nets)

**Run:**

```bash
python src/numpy_manual_stats.py
```

**What it does:**

1. **Creates synthetic 2D data** `X` with shape `(n_samples, n_features)`:

   - Think of columns as something like `[height, weight, age]` — each with different mean and std.
   - This forces you to think **column-wise**, like real ML: `(n_samples, n_features)`.

2. **Compares manual vs NumPy stats on a single feature:**

   ```text
   Means:
     manual_mean_0 = ...
     numpy_mean_0  = ...
     difference    = ...

   Variances (ddof=0):
     manual_var_0  = ...
     numpy_var_0   = ...
     difference    = ...

   Standard deviations (ddof=0):
     manual_std_0  = ...
     numpy_std_0   = ...
     difference    = ...
   ```

   If the differences are essentially zero, your manual implementation is correct.

3. **Computes column-wise stats with `axis=0`:**

   ```text
   Manual means per column: [ ... ... ... ]
   NumPy  means per column: [ ... ... ... ]
   Difference (means):      [ ~0 ~0 ~0 ]

   Manual variances per column: [...]
   NumPy  variances per column: [...]
   Difference (variances):      [...]
   ```

   This is where you *feel* that:

   - axis=0 → “aggregate over rows, get one number per column (per feature)”.

4. **Standardizes all features (z-score per column):**

   ```python
   Z, means_used, stds_used = standardize_features(X, ddof=0)
   ```

   The script prints:

   - `means_used` and `stds_used` for each feature.
   - the standardized matrix `Z`.

5. **Checks that standardized features have ~0 mean and ~1 std:**

   ```text
   Mean of each standardized column (should be close to 0):
   [ 1e-16  ... ]

   Std of each standardized column (should be close to 1):
   [ 0.99... 1.00... 1.01... ]
   ```

**Why this matters for ML:**

- In almost every ML pipeline you will see something like:

  ```python
  from sklearn.preprocessing import StandardScaler

  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X_train)
  ```

- This script shows that `StandardScaler` is basically:

  \[
  z = rac{x - \mu}{\sigma}
  \]

  applied **per feature**:

  - compute `mean_j` and `std_j` for each column `j` over the training data
  - subtract `mean_j` from each value in that column
  - divide by `std_j`

- You’re not just calling a black-box scaler — you understand:
  - how and why it works,
  - how it interacts with matrix shapes and broadcasting,
  - and why it must be fit on the training set only (to avoid data leakage).

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

- **Standardization (z-score)** → make each feature have ~0 mean and ~1 std, so models that are sensitive to scale (like linear models, gradient-based methods) behave better.

- **Gradient** → vector of partial derivatives; direction of steepest *increase*.
- **Gradient descent** → move in the opposite direction of gradient to *minimize* a function.

---

This Project 0 gives you the mental tools you need before touching “real” ML models:  
vectors, matrices, stats, standardization, and what actually happens behind `StandardScaler` and `GradientDescent`.
