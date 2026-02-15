"""
D2 – Simple MLP on MNIST with PyTorch (engineering-grade)

Adds:
- mixed precision training (AMP)
- early stopping on validation loss
- checkpointing (latest/best) + resume
- optional PyTorch profiler trace export
"""

from __future__ import annotations

import argparse
import csv
import json
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ContextManager

import torch
from torch import nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


@dataclass
class MNISTConfig:
    data_dir: Path = Path("data") / "mnist"
    batch_size: int = 128
    lr: float = 1e-3
    n_epochs: int = 8
    val_ratio: float = 0.1
    hidden_dims: tuple[int, int] = (256, 128)
    model_dir: Path = Path("models")
    output_dir: Path = Path("output") / "mnist"

    # Engineering controls
    use_mixed_precision: bool = True
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 1e-4

    enable_profiler: bool = False
    profile_steps: int = 120
    profile_dir: Path = Path("output") / "mnist" / "profiler"

    resume_from: Path | None = None
    latest_checkpoint_name: str = "mnist_mlp_latest.pt"
    best_checkpoint_name: str = "mnist_mlp_best.pt"
    final_model_name: str = "mnist_mlp.pt"


class EarlyStopping:
    """Stop training after `patience` non-improving validation-loss epochs."""

    def __init__(self, patience: int, min_delta: float, initial_best: float = float("inf")) -> None:
        if patience < 1:
            raise ValueError("patience must be >= 1")
        if min_delta < 0:
            raise ValueError("min_delta must be >= 0")

        self.patience = patience
        self.min_delta = min_delta
        self.best = initial_best
        self.bad_epochs = 0

    def step(self, value: float) -> bool:
        improved = value < (self.best - self.min_delta)
        if improved:
            self.best = value
            self.bad_epochs = 0
            return False

        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


class MLP_MNIST(nn.Module):
    def __init__(self, hidden_dims: tuple[int, int] = (256, 128)) -> None:
        super().__init__()
        h1, h2 = hidden_dims
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MNIST MLP with engineering-grade controls.")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--resume-from", type=str, default=None)

    parser.add_argument(
        "--mixed-precision",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable AMP autocast when available.",
    )
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)

    parser.add_argument(
        "--profile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable PyTorch profiler and export traces.",
    )
    parser.add_argument("--profile-steps", type=int, default=120)

    return parser.parse_args()


def prepare_dataloaders(cfg: MNISTConfig) -> tuple[DataLoader, DataLoader, DataLoader]:
    transform = transforms.Compose([transforms.ToTensor()])

    train_full = datasets.MNIST(
        root=cfg.data_dir,
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.MNIST(
        root=cfg.data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    val_size = int(len(train_full) * cfg.val_ratio)
    train_size = len(train_full) - val_size
    train_dataset, val_dataset = random_split(train_full, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

    print(
        f"[DATA] train={train_size}, val={val_size}, test={len(test_dataset)}, "
        f"batch_size={cfg.batch_size}"
    )

    return train_loader, val_loader, test_loader


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    return correct / targets.size(0)


def get_amp_settings(cfg: MNISTConfig, device: torch.device) -> tuple[bool, torch.dtype]:
    if not cfg.use_mixed_precision:
        return False, torch.float32

    if device.type == "cuda":
        return True, torch.float16

    if device.type == "cpu":
        # CPU autocast generally uses bfloat16.
        return True, torch.bfloat16

    return False, torch.float32


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    epoch: int,
    profiler: Any | None = None,
    profile_steps: int = 0,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    for batch_idx, (x, y) in enumerate(loader, start=1):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if profiler is not None and batch_idx <= profile_steps:
            profiler.step()

        batch_acc = compute_accuracy(logits, y)
        total_loss += float(loss.item())
        total_acc += batch_acc
        n_batches += 1

    avg_loss = total_loss / n_batches
    avg_acc = total_acc / n_batches

    print(f"[TRAIN] epoch={epoch:02d} | loss={avg_loss:.4f} | acc={avg_acc * 100:.2f}%")
    return avg_loss, avg_acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    stage: str,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)

        batch_acc = compute_accuracy(logits, y)
        total_loss += float(loss.item())
        total_acc += batch_acc
        n_batches += 1

    avg_loss = total_loss / n_batches
    avg_acc = total_acc / n_batches

    print(f"[{stage.upper()}] loss={avg_loss:.4f} | acc={avg_acc * 100:.2f}%")
    return avg_loss, avg_acc


def save_training_history(history: list[dict[str, float | int]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "mnist_training_metrics.json"
    csv_path = output_dir / "mnist_training_metrics.csv"

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)
    print(f"[SAVE] Saved training metrics (JSON) to {json_path}")

    if history:
        fieldnames = list(history[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for row in history:
                writer.writerow(row)
        print(f"[SAVE] Saved training metrics (CSV) to {csv_path}")


def build_checkpoint_payload(
    *,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    best_val_acc: float,
    best_val_loss: float,
    best_epoch: int,
    history: list[dict[str, float | int]],
    cfg: MNISTConfig,
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler.is_enabled() else None,
        "best_val_acc": float(best_val_acc),
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "history": history,
        "config": {
            "batch_size": cfg.batch_size,
            "lr": cfg.lr,
            "n_epochs": cfg.n_epochs,
            "hidden_dims": list(cfg.hidden_dims),
        },
    }


def load_resume_checkpoint(
    resume_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
) -> tuple[int, float, float, int, list[dict[str, float | int]]]:
    ckpt = torch.load(resume_path, map_location=device, weights_only=False)

    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    scaler_state = ckpt.get("scaler_state_dict")
    if scaler_state is not None and scaler.is_enabled():
        scaler.load_state_dict(scaler_state)

    start_epoch = int(ckpt["epoch"]) + 1
    best_val_acc = float(ckpt.get("best_val_acc", 0.0))
    best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
    best_epoch = int(ckpt.get("best_epoch", 0))

    history_raw = ckpt.get("history", [])
    history: list[dict[str, float | int]] = list(history_raw)

    print(
        f"[RESUME] Loaded checkpoint {resume_path} | "
        f"next_epoch={start_epoch}, best_val_loss={best_val_loss:.4f}, best_val_acc={best_val_acc:.4f}"
    )

    return start_epoch, best_val_acc, best_val_loss, best_epoch, history


def build_profiler(
    cfg: MNISTConfig,
    device: torch.device,
) -> ContextManager[Any]:
    if not cfg.enable_profiler:
        return nullcontext(None)

    cfg.profile_dir.mkdir(parents=True, exist_ok=True)

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    return torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(cfg.profile_dir)),
    )


def main() -> None:
    args = parse_args()

    cfg = MNISTConfig(
        batch_size=args.batch_size,
        lr=args.lr,
        n_epochs=args.epochs,
        use_mixed_precision=bool(args.mixed_precision),
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        enable_profiler=bool(args.profile),
        profile_steps=args.profile_steps,
        resume_from=Path(args.resume_from) if args.resume_from else None,
    )

    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    use_amp, amp_dtype = get_amp_settings(cfg, device)
    scaler = GradScaler(enabled=(use_amp and device.type == "cuda"))
    print(
        f"[AMP] enabled={use_amp} | dtype={amp_dtype} | "
        f"grad_scaler_enabled={scaler.is_enabled()}"
    )

    train_loader, val_loader, test_loader = prepare_dataloaders(cfg)

    model = MLP_MNIST(hidden_dims=cfg.hidden_dims).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    latest_ckpt_path = cfg.model_dir / cfg.latest_checkpoint_name
    best_ckpt_path = cfg.model_dir / cfg.best_checkpoint_name

    start_epoch = 1
    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_epoch = 0
    history: list[dict[str, float | int]] = []

    if cfg.resume_from is not None:
        if not cfg.resume_from.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {cfg.resume_from}")

        start_epoch, best_val_acc, best_val_loss, best_epoch, history = load_resume_checkpoint(
            cfg.resume_from,
            model,
            optimizer,
            scaler,
            device,
        )

    early_stopper = EarlyStopping(
        patience=cfg.early_stopping_patience,
        min_delta=cfg.early_stopping_min_delta,
        initial_best=best_val_loss,
    )

    with build_profiler(cfg, device) as profiler:
        for epoch in range(start_epoch, cfg.n_epochs + 1):
            train_loss, train_acc = train_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                epoch=epoch,
                profiler=profiler,
                profile_steps=cfg.profile_steps,
            )

            val_loss, val_acc = evaluate(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                stage="val",
            )

            history.append(
                {
                    "epoch": epoch,
                    "train_loss": float(train_loss),
                    "train_acc": float(train_acc),
                    "val_loss": float(val_loss),
                    "val_acc": float(val_acc),
                }
            )

            payload = build_checkpoint_payload(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                best_val_acc=best_val_acc,
                best_val_loss=best_val_loss,
                best_epoch=best_epoch,
                history=history,
                cfg=cfg,
            )
            torch.save(payload, latest_ckpt_path)

            if val_loss < (best_val_loss - cfg.early_stopping_min_delta):
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_epoch = epoch

                best_payload = build_checkpoint_payload(
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    best_val_acc=best_val_acc,
                    best_val_loss=best_val_loss,
                    best_epoch=best_epoch,
                    history=history,
                    cfg=cfg,
                )
                torch.save(best_payload, best_ckpt_path)

            should_stop = early_stopper.step(val_loss)
            if should_stop:
                print(
                    "[EARLY-STOP] "
                    f"Stopped at epoch={epoch}; best_epoch={best_epoch}; "
                    f"best_val_loss={best_val_loss:.4f}"
                )
                break

    if cfg.enable_profiler:
        print(f"[PROFILE] Trace files written to {cfg.profile_dir}")

    if best_ckpt_path.exists():
        best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt["model_state_dict"])

    print(
        f"[INFO] Best validation checkpoint: epoch={best_epoch}, "
        f"val_loss={best_val_loss:.4f}, val_acc={best_val_acc * 100:.2f}%"
    )

    _, test_acc = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        stage="test",
    )
    print(f"[RESULT] Test accuracy={test_acc * 100:.2f}%")

    final_model_path = cfg.model_dir / cfg.final_model_name
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "model": {
                    "input_dim": 28 * 28,
                    "hidden_dims": list(cfg.hidden_dims),
                    "num_classes": 10,
                },
                "train": {
                    "batch_size": cfg.batch_size,
                    "lr": cfg.lr,
                    "n_epochs": cfg.n_epochs,
                    "use_mixed_precision": use_amp,
                },
            },
            "best_val_acc": float(best_val_acc),
            "best_val_loss": float(best_val_loss),
            "best_epoch": int(best_epoch),
            "test_acc": float(test_acc),
        },
        final_model_path,
    )
    print(f"[SAVE] Final model checkpoint saved to {final_model_path}")
    print(f"[SAVE] Latest checkpoint: {latest_ckpt_path}")
    if best_ckpt_path.exists():
        print(f"[SAVE] Best checkpoint:   {best_ckpt_path}")

    save_training_history(history, cfg.output_dir)


if __name__ == "__main__":
    main()
