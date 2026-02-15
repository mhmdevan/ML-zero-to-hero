from __future__ import annotations

import pytest


def test_make_synthetic_data_shapes() -> None:
    torch = pytest.importorskip("torch")
    module = pytest.importorskip("src.torch_autograd_gd")

    cfg = module.GDConfig(n_samples=12)
    x, y = module.make_synthetic_data(cfg, device=torch.device("cpu"))

    assert x.shape == (12, 1)
    assert y.shape == (12, 1)


def test_run_nn_linear_gd_rejects_unknown_optimizer() -> None:
    torch = pytest.importorskip("torch")
    module = pytest.importorskip("src.torch_autograd_gd")

    cfg = module.GDConfig(n_steps=2, n_samples=8)
    with pytest.raises(ValueError):
        module.run_nn_linear_gd(cfg, device=torch.device("cpu"), optimizer_name="unknown")
