# Numerical Stability

This chapter upgrades Project 0 from "runs on my machine" scripts to safer ML fundamentals.

## 1) Overflow in Softmax / Log-Sum-Exp

When logits are large, direct exponentiation can overflow:

- Naive: `exp([1000, 1001, 1002]) -> inf`
- Result: `softmax` and `log(sum(exp(.)))` can become `nan` / `inf`.

Stable transform:

1. Let `m = max(logits)`
2. Compute on shifted logits: `logits - m`

Formulas:

- `logsumexp(x) = m + log(sum(exp(x - m)))`
- `softmax(x)_i = exp(x_i - m) / sum_j exp(x_j - m)`

Implemented in:

- `src/numerical_stability.py`
  - `stable_log_sum_exp`
  - `stable_softmax`

## 2) Gradient Explosion and Clipping

Large gradients create unstable updates:

- loss spikes
- parameter jumps
- divergence (`nan`/`inf`)

Norm clipping rescales gradients only when needed:

- if `||g||_2 <= max_norm`: keep as-is
- else: `g <- g * (max_norm / ||g||_2)`

Implemented in:

- `src/numerical_stability.py`
  - `clip_gradients_numpy`
  - `clip_gradients_torch`

## 3) Practical Policy for This Project

Use these defaults in educational scripts:

- Prefer stable softmax/log-sum-exp versions.
- Add gradient clipping in training loops where gradients can spike.
- Keep checks in tests (`tests/test_numerical_stability.py`) so regressions are caught early.

## 4) Run the Demo

```bash
python -m src.numerical_stability
```

Expected output shows:

- naive log-sum-exp becomes non-finite
- stable log-sum-exp remains finite
- clipped gradient norm is capped to `max_norm`
