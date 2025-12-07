# 🧠 ML Project 0 – Python, Math & PyTorch Foundations

End-to-end **foundations for Machine Learning** in pure Python and PyTorch, written as small, focused scripts (no notebooks).  
Goal: not just “run ML”, but **understand the building blocks** you will keep reusing in all later projects.

This repo has two big parts:

- **Classic foundations (NumPy + math)**  
  Vectors, matrices, manual statistics, standardization, 1D gradient descent.
- **Deep Learning fundamentals (PyTorch)**  
  Autograd demo, an MLP on MNIST, and an LSTM on a synthetic time series with proper baselines & metrics.

Everything is organized to be **resume-friendly** and easy to explain in an interview:

- What was the **problem**?
- What **data** did you use?
- Which **packages/models** did you try and why?
- What **metrics** did you track?
- What went wrong (**overfitting / leakage / baselines**) and how did you fix it?

---

## 1. Tech Stack

**Languages & Core libs**

- Python 3.11+
- NumPy – vectors, matrices, manual stats, broadcasting
- Matplotlib – basic plots for intuition

**Deep Learning**

- PyTorch – tensors, autograd, optimizers (SGD/Adam)
- Torchvision – MNIST dataset & transforms

**Other**

- `argparse`, `pathlib`, `json` – for CLI-style scripts and saving metrics

No notebooks – everything is **script-first**, to look and feel like real code in a codebase.

---

## 2. Project Structure

```text
0-foundations/
  ├─ src/
  │   ├─ numpy_basics.py               # ndarray, dot, matmul, transpose, broadcasting
  │   ├─ linear_algebra_demo.py        # solving Ax=b, determinant, inverse, eigenvalues/vectors
  │   ├─ stats_basics.py               # mean, variance, covariance, correlation, normal dist + plots
  │   ├─ gradient_descent_demo.py      # 1D gradient descent on f(w) = (w - 3)^2
  │   ├─ numpy_manual_stats.py         # manual mean/var/std + standardization (1D & 2D)
  │   ├─ torch_gd_autograd.py          # 1D gradient descent re-implemented in PyTorch with autograd + optim
  │   ├─ torch_mlp_mnist.py            # MLP on MNIST: training loop + metrics saving
  │   ├─ torch_mnist_inference.py      # inference on a single MNIST image (test set or custom PNG)
  │   ├─ save_mnist_sample.py          # helper to save a test MNIST digit as PNG (e.g. digit_42.png)
  │   └─ torch_lstm_timeseries.py      # LSTM on synthetic time-series + naive baseline + metrics/plot
  │
  ├─ output/
  │   ├─ plots/                        # classic stats / GD plots (heights, weights, GD curve, …)
  │   ├─ mnist/                        # MNIST training metrics (loss/acc vs epoch)
  │   └─ timeseries/                   # LSTM vs naive baseline CSV + plot of predictions vs true series
  │
  ├─ models/
  │   └─ mnist_mlp.pt                  # trained MLP checkpoint for MNIST inference (ignored by git)
  │
  ├─ data/
  │   └─ (MNIST will be downloaded here automatically by torchvision)
  │
  ├─ requirements.txt
  └─ README.md
```

> **Note:** large/binary artifacts (full datasets, `.pt` checkpoints, etc.) are ignored via `.gitignore`.  
> Only small, human-readable outputs (e.g. plots) are meant to be committed.

---

## 3. Setup

From the `0-foundations/` directory:

```bash
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Everything runs on CPU – no GPU is required.

---

## 4. Classic Foundations (NumPy + Math)

These scripts are **not toys** – they are the mental model behind everything you will do later with scikit-learn / PyTorch.

### 4.1 `numpy_basics.py` – Arrays, dot, matmul, broadcasting

**Problem:** Get comfortable with `ndarray` shapes and core operations.

**What it covers**

- Creating 1D and 2D arrays
- Inspecting shapes and dtypes
- Dot product vs manual sum of element-wise products
- Matrix multiplication with `@`
- Transpose `.T`
- Broadcasting rules when adding row/column vectors to matrices

**Run:**

```bash
python -m src.numpy_basics
```

This prints shapes and example computations and saves small matrix dumps (if configured) into `output/` (or `plots/` in older versions).

---

### 4.2 `linear_algebra_demo.py` – Solving Ax = b, det, inverse, eigen

**Problem:** Build intuition for linear systems and transformations.

**What it covers**

- Constructing a 2×2 (or 3×3) matrix `A` and vector `b`
- Solving `Ax = b` via `np.linalg.solve`
- Computing determinant `np.linalg.det(A)`
- Inverse `A_inv = np.linalg.inv(A)` and checking `A @ A_inv ≈ I`
- Eigenvalues & eigenvectors of `A`, checking `A @ v ≈ λ v`

**Run:**

```bash
python -m src.linear_algebra_demo
```

This script prints all intermediate results with clear labels so you can see the linear algebra steps.

---

### 4.3 `stats_basics.py` – Basic statistics + plots

**Problem:** Understand basic descriptive statistics on synthetic data.

**What it covers**

- Sampling synthetic heights and weights from (approx.) normal distributions
- Computing mean, variance, standard deviation
- Covariance & correlation between height and weight
- Plotting:
  - histogram of heights
  - histogram of weights
  - scatter plot of height vs weight

**Run:**

```bash
python -m src.stats_basics
```

**Outputs (saved into `output/plots/`):**

- `hist_heights.png` – height distribution
- `hist_weights.png` – weight distribution
- `scatter_height_vs_weight.png` – correlation visualized

These plots are perfect to embed into README or slides when explaining basic stats.

---

### 4.4 `gradient_descent_demo.py` – 1D Gradient Descent by hand

**Problem:** See gradient descent as numbers moving, not just a formula.

**Objective function**

- \( f(w) = (w - 3)^2 \)
- derivative \( f'(w) = 2(w - 3) \)

**What it does**

- Initialize `w` at some value (e.g. `w_init = -5`)
- Run a loop:

  ```python
  w = w - lr * grad(w)
  ```

- Log the value of `w` and `f(w)` across iterations
- Plot the curve + the points of gradient descent moving towards the minimum at `w = 3`

**Run:**

```bash
python -m src.gradient_descent_demo
```

**Output:**

- `output/plots/gradient_descent_1d.png` – red dots walking down a quadratic bowl towards `w = 3`.

This is the mental template for every later optimization (linear regression, logistic regression, neural nets).

---

### 4.5 `numpy_manual_stats.py` – Manual stats & standardization (Project 0.2)

**Problem:** Stop treating `np.mean`, `np.var`, `StandardScaler` as “magic”.

**Key functions implemented using only basic NumPy:**

- `manual_mean(x, axis=None)`
- `manual_variance(x, axis=None, ddof=0)`
- `manual_std(x, axis=None, ddof=0)`
- `standardize_1d(x)` – z-score for 1D
- `standardize_features(X)` – column-wise standardization for 2D `(n_samples, n_features)`

**What it demonstrates**

1. Creates a synthetic matrix `X` with shape `(n_samples, n_features)`.
2. Compares manual stats vs NumPy built-ins:

   ```text
   Means:
     manual_mean_0 = ...
     numpy_mean_0  = ...
     difference    = ...
   ```

3. Shows how `axis=0` vs `axis=1` behave.
4. Applies standardization per column and checks that:
   - standardized columns have mean ≈ 0
   - and std ≈ 1

**Run:**

```bash
python -m src.numpy_manual_stats
```

**Why it matters**

- Later, when you see:

  ```python
  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X_train)
  ```

  you know it’s just:

  \[ z = \frac{x - \mu}{\sigma} \]

  applied **per feature**, with means/stds computed on **training data only** (to avoid data leakage).

---

## 5. Deep Learning Fundamentals (PyTorch)

Here we **re-build** some of the above concepts using PyTorch:

- tensors instead of ndarrays,
- autograd instead of manual derivatives,
- simple models (MLP, LSTM) with proper baselines & metrics.

### 5.1 `torch_gd_autograd.py` – Gradient Descent with autograd + optim

**Problem:** Show how PyTorch’s autograd + optimizers reproduce manual GD.

**What it does**

- Defines the same 1D function: \( f(w) = (w - 3)^2 \)
- Creates a scalar parameter `w = torch.tensor(..., requires_grad=True)`
- In each step:
  - calls `loss.backward()` to compute `w.grad`
  - calls `optimizer.step()` (e.g. SGD)
  - zeros gradients with `optimizer.zero_grad()`

**Run:**

```bash
python -m src.torch_gd_autograd
```

**Why this is important**

- It connects the dots directly to `gradient_descent_demo.py`:
  - Same function and idea,
  - But now PyTorch does the derivative and parameter updates.
- This is the exact pattern you’ll reuse for every deep learning model later.

---

### 5.2 `torch_mlp_mnist.py` – MLP on MNIST

**Problem:** Build and train a simple neural network on a real image dataset.

**Model**

- Fully-connected MLP:

  - input: 28×28 = 784 features
  - hidden layers: e.g. `[256, 128]` with ReLU
  - output: 10 logits (digits 0–9)

- Loss: `nn.CrossEntropyLoss`
- Optimizer: `Adam`

**Data**

- MNIST from `torchvision.datasets.MNIST`
  - Train split (60k images)
  - Test split (10k images)

**What the script does**

1. Downloads MNIST to `data/` (if not present).
2. Creates `DataLoader`s for train & test.
3. Runs a training loop for N epochs (configurable).
4. On each epoch:
   - logs **train loss**
   - runs evaluation on test set: **test loss**, **test accuracy**
5. Saves metrics into `output/mnist/` as JSON/CSV (e.g. one row per epoch).
6. Saves the final model checkpoint as `models/mnist_mlp.pt` (ignored by git).

**Run:**

```bash
python -m src.torch_mlp_mnist
```

**Sample result (CPU, a few epochs)**

- With a simple 2–3 layer MLP and a handful of epochs, you typically get:
  - Test accuracy ≈ 97–98%
  - Test loss steadily decreasing

These metrics are saved so you can embed a **training curve** screenshot in your GitHub README.

**Design choices / trade-offs**

- MLP instead of CNN:
  - simpler to read & explain in a foundations project,
  - good enough to show how image data is flattened and fed to a dense network.
- Adam.optimizer:
  - converges faster and more robust than plain SGD for this toy run,
  - easier to show quick progress on CPU.

---

### 5.3 `torch_mnist_inference.py` – Single-image inference

**Problem:** Not only “train a model”, but also show a clean **inference path**.

**What it does**

- Loads `models/mnist_mlp.pt` and its config (input dim, hidden sizes, num classes).
- Accepts either:
  - an index into the MNIST test set (`--index 42`), or
  - a custom PNG path (`--image-path path/to/digit.png`).

- Preprocesses the image to `[1, 1, 28, 28]` (grayscale, resized, normalized to [0,1]).
- Runs the MLP and prints:
  - `predicted_digit`
  - `predicted_prob` (max softmax probability)
  - full probability distribution over 10 digits.

**Run with test sample:**

```bash
python -m src.torch_mnist_inference --index 42
```

**Run with custom PNG:**
First, use the helper script to export a sample:

```bash
python -m src.save_mnist_sample
# -> saves e.g. src/digit_42.png
```

Then:

```bash
python -m src.torch_mnist_inference --image-path src/digit_42.png
```

This mirrors what you did in the spam project (`predict_spam.py`), but now for vision.

---

### 5.4 `torch_lstm_timeseries.py` – LSTM vs Naive baseline on synthetic sales

**Problem:** Show how a sequence model behaves on a time series, and **compare it to a naive baseline** (not just raw loss).

**Data**

- A synthetic “monthly sales” time series of length 360, generated as:

  - seasonality (e.g. yearly sinus pattern),
  - trend + noise,
  - values scaled to a realistic range (e.g. 70–160).

- Train / test split:
  - Train: first 288 points
  - Test: last 72 points (respecting time order – no shuffle).

- We fit a `StandardScaler` **only on the train series** and reuse it for test, to avoid **data leakage**.

**Input windows**

- We build sliding windows of:
  - `input_window = 30` time steps → model sees 30 past points,
  - `horizon = 1` → predicts the next 1 step.
- This creates:
  - `(train_samples, 30)` for train
  - `(test_samples, 30)` for test

**Model**

- 1-layer LSTM:
  - `input_size = 1` (univariate series)
  - `hidden_size` (configurable, e.g. 32)
  - `num_layers = 1`
- Followed by a Linear head to predict one scalar.
- Loss: MSE
- Optimizer: Adam

**Baseline**

- Naive “last value” baseline:
  - prediction for `t+1` = last observed value at `t`.
- Baseline MSE computed on the **same test horizon**, to make comparison fair.

**Training run**

```bash
python -m src.torch_lstm_timeseries
```

Example log (with the improved version):

```text
[INFO] Generated synthetic sales series of length 360 (min=74.65, max=158.35)
[DATA] Train series length=288, Test series length=72
[SCALE] Fitted StandardScaler on train series (mean=110.619, std=14.336)
[DATA] Sliding windows: X.shape=(330, 30), y.shape=(330, 1), input_window=30, horizon=1
[TRAIN] epoch=120 | train_loss ~ 0.34
[TEST] LSTM MSE=178.63
[BASELINE] Naive last-value MSE=262.86
[RESULT] LSTM MSE=178.63 vs Naive baseline MSE=262.86 (diff=-84.23)
```

**Outputs**

- `output/timeseries/lstm_predictions_vs_true.csv`  
  Contains actual vs predicted values for the test range.
- `output/timeseries/lstm_predictions_plot.png`  
  A line plot showing:
  - full series,
  - train/test split,
  - LSTM predictions vs true values on test.

**Why this matters in an interview**

- You don’t just “train an LSTM” – you:
  - build a realistic baseline,
  - avoid leakage by fitting the scaler on train only,
  - use proper **time-based split** instead of random,
  - compare and report both metrics clearly.

---

## 6. Design Decisions, Metrics & Typical Pitfalls

### 6.1 Data shapes and mental model

Across all scripts, data is consistently shaped as:

- **Tabular / vector data:** `(n_samples, n_features)`
- **Time series:** `(n_samples, input_window, 1)` for LSTM
- **Images:** `[batch, channels, height, width]` = `[B, 1, 28, 28]` for MNIST

This consistency makes it easy to move from pure NumPy to PyTorch and later to scikit-learn pipelines.

---

### 6.2 Metrics

- **Gradient descent demos:** no explicit numeric metric, but you see:
  - convergence of `w` to the minimum,
  - decreasing `f(w)` visually and in logs.

- **MNIST MLP:**
  - Train loss per epoch
  - Test loss per epoch
  - Test accuracy per epoch
  - Metrics are saved into `output/mnist/…` so you can plot loss/accuracy curves.

- **LSTM time series:**
  - MSE on **scaled** predictions, rescaled back to original units for human-readable interpretation.
  - Naive baseline MSE for comparison.
  - Final summary line:

    ```text
    [SUMMARY] LSTM MSE=..., Naive MSE=...
    ```

  so you can quickly see if the model is genuinely useful.

---

### 6.3 Data leakage, overfitting, drift – how they appear here

Even in a “foundations” repo, these concepts matter – and you can already talk about them:

- **Data leakage**
  - MNIST: standard train/test split from torchvision, no leakage.
  - Time series:
    - Train/test split respects time order (no shuffle).
    - StandardScaler is **fit on train only**, then applied to test.
    - This is explicitly logged and commented in code.

- **Overfitting**
  - MNIST MLP:
    - Model is intentionally small (2–3 layers) and trained for a moderate number of epochs.
    - You can inspect loss/accuracy curves; if train accuracy >> test accuracy, you can talk about reducing capacity or adding regularization.
  - LSTM:
    - We started with a shorter series and small window, saw poor performance vs baseline,
    - then moved to a longer series and more reasonable window size (30), giving a more meaningful comparison.

- **Drift**
  - For MNIST, we assume stationarity (data distribution doesn’t change).
  - For the synthetic time series, we control the generative process.
  - In a README / discussion, you can already say:
    - “In real production, I would monitor distributions of the time series (level, variance, seasonality) and retrain the model when drift is detected.”

This repo doesn’t try to solve drift fully, but it introduces the right **way of thinking**.

---

## 7. How This Project Fits Into the Bigger ML Roadmap

This is **Project 0** in a larger ML roadmap (Sales regression, Spam classifier, User clustering, …).  
What it gives you:

- Confidence with `ndarray` and tensor shapes.
- Intuition for gradient descent and standardization.
- A first full PyTorch training loop + inference path on MNIST.
- A first sequence model with a real baseline on a time series.

Later projects build on this:

- **Tabular regression/classification** (scikit-learn)
- **Text classification** (TF-IDF + Linear models)
- **User clustering** (K-Means + PCA)
- And eventually **full MLOps v2** projects (DuckDB, MLflow, DVC, Airflow).

---

## 8. License

```text
MIT License

Copyright (c) 2025 Mohammad Eslamnia
...
```
