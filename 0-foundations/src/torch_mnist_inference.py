"""
torch_mnist_inference.py

Simple inference script for the MNIST MLP model trained in `torch_mlp_mnist.py`.

Usage:
    python -m src.torch_mnist_inference
    python -m src.torch_mnist_inference --index 42
    python -m src.torch_mnist_inference --image-path path/to/digit.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from PIL import Image
from torchvision import datasets, transforms


class MNISTMLP(nn.Module):
    """
    Simple MLP for MNIST classification.
    Same architecture as in `torch_mlp_mnist.py`.
    """

    def __init__(
        self,
        input_dim: int = 28 * 28,
        hidden_dims: tuple[int, ...] = (256, 128),
        num_classes: int = 10,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is expected to be [batch_size, 1, 28, 28]
        x = x.view(x.size(0), -1)
        return self.net(x)


def load_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[MNISTMLP, dict[str, Any]]:
    """
    Load a checkpoint file and reconstruct the model.

    IMPORTANT (PyTorch 2.6+):
    - torch.load(...) now defaults to weights_only=True.
    - Our checkpoint contains Python objects in config.
    - For our own trusted checkpoint, we set weights_only=False.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        cfg = ckpt.get("config", {})
    else:
        state_dict = ckpt
        cfg = {}

    model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}

    input_dim = int(model_cfg.get("input_dim", 28 * 28))
    hidden_dims_raw = model_cfg.get("hidden_dims", [256, 128])
    hidden_dims = tuple(int(hidden_dim) for hidden_dim in hidden_dims_raw)
    num_classes = int(model_cfg.get("num_classes", 10))

    model = MNISTMLP(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        num_classes=num_classes,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, (cfg if isinstance(cfg, dict) else {})


def load_mnist_test_dataset(data_dir: Path) -> datasets.MNIST:
    transform = transforms.Compose([transforms.ToTensor()])

    test_dataset = datasets.MNIST(
        root=str(data_dir),
        train=False,
        download=True,
        transform=transform,
    )
    return test_dataset


def load_single_image(image_path: Path, device: torch.device) -> torch.Tensor:
    """
    Load a single PNG/JPEG image from disk and convert to [1, 1, 28, 28].
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = Image.open(image_path).convert("L").resize((28, 28))
    tensor = transforms.ToTensor()(img)  # [1, 28, 28], values in [0, 1]
    return tensor.unsqueeze(0).to(device)


@torch.no_grad()
def predict_digit(model: MNISTMLP, x: torch.Tensor) -> dict[str, Any]:
    """
    Given a single image batch [1, 1, 28, 28], return prediction and probs.
    """
    logits = model(x)  # [1, 10]
    probs = torch.softmax(logits, dim=1)  # [1, 10]
    prob_values, pred_indices = torch.max(probs, dim=1)

    predicted_digit = int(pred_indices.item())
    predicted_prob = float(prob_values.item())
    probs_list = [float(value) for value in probs.squeeze(0).cpu().tolist()]

    return {
        "predicted_digit": predicted_digit,
        "predicted_prob": predicted_prob,
        "probabilities": probs_list,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MNIST MLP inference script (single image).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/mnist_mlp.pt",
        help="Path to the trained MNIST MLP checkpoint (.pt file).",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help=(
            "Index of the sample in the MNIST test set to predict. "
            "If provided, --image-path is ignored."
        ),
    )
    parser.add_argument(
        "--image-path",
        type=str,
        default=None,
        help=(
            "Path to a custom image (28x28 or resized). "
            "If not provided, a sample from MNIST test set is used."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    project_root = Path(__file__).resolve().parent.parent
    checkpoint_path = project_root / args.checkpoint
    data_dir = project_root / "data"

    model, cfg_dict = load_checkpoint(checkpoint_path, device=device)
    print(f"[INFO] Loaded checkpoint from {checkpoint_path}")
    if cfg_dict:
        print(f"[INFO] Model config keys: {list(cfg_dict.keys())}")

    if args.image_path:
        if args.index is not None:
            print("[WARN] --index is ignored because --image-path was provided.")
        source = f"Custom image: {args.image_path}"
        x = load_single_image(Path(args.image_path), device=device)
        true_label: int | None = None
    else:
        test_dataset = load_mnist_test_dataset(data_dir)
        if args.index is None:
            idx = torch.randint(low=0, high=len(test_dataset), size=(1,)).item()
        else:
            idx = args.index
            if idx < 0 or idx >= len(test_dataset):
                raise IndexError(
                    f"Index {idx} is out of range for MNIST test set (0..{len(test_dataset)-1})."
                )

        x_img, y_label = test_dataset[idx]
        x = x_img.unsqueeze(0).to(device)
        true_label = int(y_label)
        source = f"MNIST test set index={idx}"

    result = predict_digit(model, x)

    print()
    print("[INFERENCE RESULT]")
    print(f"source: {source}")
    if true_label is not None:
        print(f"true_label: {true_label}")
    print(f"predicted_digit: {result['predicted_digit']}")
    print(f"predicted_prob: {result['predicted_prob']:.4f}")

    probs = result["probabilities"]
    top3 = sorted(enumerate(probs), key=lambda item: item[1], reverse=True)[:3]
    print("top3 probabilities:")
    for digit, prob in top3:
        print(f"  - digit={digit} prob={prob:.4f}")


if __name__ == "__main__":
    main()
