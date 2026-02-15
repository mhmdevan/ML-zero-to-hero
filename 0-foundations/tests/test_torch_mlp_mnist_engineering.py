from __future__ import annotations

from pathlib import Path

import pytest


def _load_module():
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    return pytest.importorskip("src.torch_mlp_mnist")


def test_early_stopping_patience_behavior() -> None:
    module = _load_module()

    stopper = module.EarlyStopping(patience=2, min_delta=0.0)
    assert stopper.step(1.0) is False
    assert stopper.step(1.1) is False
    assert stopper.step(1.2) is True


def test_load_resume_checkpoint_restores_state(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    module = _load_module()

    model = module.MLP_MNIST(hidden_dims=(16, 8))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = module.GradScaler(enabled=False)

    cfg = module.MNISTConfig(n_epochs=3, hidden_dims=(16, 8))
    history = [{"epoch": 1, "train_loss": 1.0, "train_acc": 0.5, "val_loss": 0.9, "val_acc": 0.6}]

    payload = module.build_checkpoint_payload(
        epoch=1,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        best_val_acc=0.6,
        best_val_loss=0.9,
        best_epoch=1,
        history=history,
        cfg=cfg,
    )

    checkpoint_path = Path(tmp_path) / "resume.pt"
    torch.save(payload, checkpoint_path)

    start_epoch, best_acc, best_loss, best_epoch, restored_history = module.load_resume_checkpoint(
        resume_path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        device=torch.device("cpu"),
    )

    assert start_epoch == 2
    assert best_acc == pytest.approx(0.6)
    assert best_loss == pytest.approx(0.9)
    assert best_epoch == 1
    assert len(restored_history) == 1
