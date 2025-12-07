"""
D2 – Simple MLP on MNIST with PyTorch

- 2–3 Linear layers + ReLU
- Trained with CrossEntropyLoss + Adam
- Target: test accuracy > 95%
- Logs train/val metrics per epoch and saves them to:
    output/mnist/mnist_training_metrics.json
    output/mnist/mnist_training_metrics.csv
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


# -----------------------------
# Config
# -----------------------------


@dataclass
class MNISTConfig:
    data_dir: Path = Path("data") / "mnist"
    batch_size: int = 128
    lr: float = 1e-3
    n_epochs: int = 5
    val_ratio: float = 0.1
    hidden_dims: Tuple[int, int] = (256, 128)
    model_dir: Path = Path("models")
    output_dir: Path = Path("output") / "mnist"  # for metrics, plots, etc.


# -----------------------------
# Model
# -----------------------------


class MLP_MNIST(nn.Module):
    def __init__(self, hidden_dims: Tuple[int, int] = (256, 128)):
        super().__init__()
        h1, h2 = hidden_dims
        self.net = nn.Sequential(
            nn.Flatten(),              # 28x28 -> 784
            nn.Linear(28 * 28, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, 10),        # 10 classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -----------------------------
# Data
# -----------------------------


def prepare_dataloaders(cfg: MNISTConfig) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Download MNIST (if needed) and create train/val/test loaders.
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # [0,1]
        ]
    )

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


# -----------------------------
# Training / Evaluation
# -----------------------------


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    return correct / targets.size(0)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        batch_acc = compute_accuracy(logits, y)
        total_loss += loss.item()
        total_acc += batch_acc
        n_batches += 1

    avg_loss = total_loss / n_batches
    avg_acc = total_acc / n_batches

    print(
        f"[TRAIN] epoch={epoch:02d} | loss={avg_loss:.4f} | acc={avg_acc * 100:.2f}%"
    )
    return avg_loss, avg_acc


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    stage: str,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            batch_acc = compute_accuracy(logits, y)
            total_loss += loss.item()
            total_acc += batch_acc
            n_batches += 1

    avg_loss = total_loss / n_batches
    avg_acc = total_acc / n_batches

    print(
        f"[{stage.upper()}] loss={avg_loss:.4f} | acc={avg_acc * 100:.2f}%"
    )
    return avg_loss, avg_acc


def save_training_history(
    history: list[dict],
    output_dir: Path,
) -> None:
    """
    Save training/validation metrics history to JSON and CSV files.

    history: list of dicts like:
      {
        "epoch": 1,
        "train_loss": ...,
        "train_acc": ...,
        "val_loss": ...,
        "val_acc": ...
      }
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "mnist_training_metrics.json"
    csv_path = output_dir / "mnist_training_metrics.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"[SAVE] Saved training metrics (JSON) to {json_path}")

    if history:
        fieldnames = list(history[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in history:
                writer.writerow(row)
        print(f"[SAVE] Saved training metrics (CSV) to {csv_path}")


# -----------------------------
# Main
# -----------------------------


def main() -> None:
    cfg = MNISTConfig()
    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    train_loader, val_loader, test_loader = prepare_dataloaders(cfg)

    model = MLP_MNIST(hidden_dims=cfg.hidden_dims).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    best_val_acc = 0.0
    best_state_dict = None

    history: list[dict] = []

    for epoch in range(1, cfg.n_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device, stage="val"
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

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state_dict = model.state_dict()

    # Restore best model on validation
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    print(f"[INFO] Best val acc={best_val_acc * 100:.2f}%")

    # Final test accuracy
    _, test_acc = evaluate(model, test_loader, criterion, device, stage="test")
    print(f"[RESULT] Test accuracy={test_acc * 100:.2f}%")

    # Save model checkpoint
    model_path = cfg.model_dir / "mnist_mlp.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": cfg.__dict__,
            "best_val_acc": float(best_val_acc),
            "test_acc": float(test_acc),
        },
        model_path,
    )
    print(f"[SAVE] Saved trained MNIST MLP to {model_path}")

    # Save training history
    save_training_history(history, cfg.output_dir)


if __name__ == "__main__":
    main()
