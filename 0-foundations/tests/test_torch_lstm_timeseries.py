from __future__ import annotations

import numpy as np
import pytest


def _load_module():
    pytest.importorskip("torch")
    pytest.importorskip("pandas")
    pytest.importorskip("matplotlib")
    pytest.importorskip("sklearn")
    return pytest.importorskip("src.torch_lstm_timeseries")


def test_create_sliding_windows_shapes() -> None:
    module = _load_module()

    series = np.linspace(0.0, 1.0, num=20, dtype=np.float32)
    x, y, idx = module.create_sliding_windows(series, input_window=5, horizon=1)

    assert x.shape == (15, 5)
    assert y.shape == (15, 1)
    assert idx.shape == (15,)


def test_create_sliding_windows_rejects_too_short_series() -> None:
    module = _load_module()

    series = np.linspace(0.0, 1.0, num=5, dtype=np.float32)
    with pytest.raises(ValueError):
        module.create_sliding_windows(series, input_window=5, horizon=2)


def test_train_lstm_returns_model() -> None:
    torch = pytest.importorskip("torch")
    module = _load_module()

    rng = np.random.default_rng(seed=0)
    x_train = rng.normal(size=(24, 6)).astype(np.float32)
    y_train = rng.normal(size=(24, 1)).astype(np.float32)

    model_cfg = module.ModelConfig(hidden_size=8, num_layers=1, dropout=0.0)
    train_cfg = module.TrainConfig(batch_size=8, num_epochs=1, lr=1e-3)

    model = module.train_lstm(
        X_train=x_train,
        y_train=y_train,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        device=torch.device("cpu"),
    )

    x_tensor = torch.from_numpy(x_train[:2]).float().unsqueeze(-1)
    pred = model(x_tensor)
    assert pred.shape == (2, 1)
