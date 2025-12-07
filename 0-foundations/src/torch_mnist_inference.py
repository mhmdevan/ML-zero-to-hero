"""
torch_mnist_inference.py

Simple inference script for the MNIST MLP model trained in `torch_mlp_mnist.py`.

Usage:
    python -m src.torch_mnist_inference
    python -m src.torch_mnist_inference --index 42
    python -m src.torch_mnist_inference --image-path path/to/digit.png
"""

import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from PIL import Image


# ---------------------------------------------------------
# 1. Model definition (must match training script)
# ---------------------------------------------------------


class MNISTMLP(nn.Module):
    """
    Simple MLP for MNIST classification.
    Same architecture as in `torch_mlp_mnist.py`.
    """

    def __init__(
        self,
        input_dim: int = 28 * 28,
        hidden_dims: Tuple[int, ...] = (256, 128),
        num_classes: int = 10,
    ) -> None:
        super().__init__()

        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is expected to be [batch_size, 1, 28, 28]
        # flatten to [batch_size, 784]
        x = x.view(x.size(0), -1)
        return self.net(x)


# ---------------------------------------------------------
# 2. Checkpoint loading
# ---------------------------------------------------------


def load_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[MNISTMLP, Dict[str, Any]]:
    """
    Load a checkpoint file and reconstruct the model.

    IMPORTANT (PyTorch 2.6+):
    - torch.load(...) now defaults to weights_only=True.
    - Our checkpoint contains Python objects (e.g. pathlib.PosixPath) inside the config.
    - For *our own* trusted checkpoint, we explicitly set weights_only=False
      to allow full unpickling.

    If you got this file from an untrusted source, DO NOT DO THIS.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # We trust this checkpoint because it was produced by our own training script.
    # So we disable weights_only to allow full unpickling of config objects.
    ckpt = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,  # <-- key fix for PyTorch 2.6+
    )

    # Two possible formats:
    # 1) { "model_state_dict": ..., "config": {...} }
    # 2) raw state_dict
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        cfg = ckpt.get("config", {})
    else:
        state_dict = ckpt
        cfg = {}

    # Try to reconstruct model hyperparameters from config, if available
    model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}

    input_dim = int(model_cfg.get("input_dim", 28 * 28))
    hidden_dims_raw = model_cfg.get("hidden_dims", [256, 128])
    # ensure tuple of ints
    hidden_dims = tuple(int(h) for h in hidden_dims_raw)
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


# ---------------------------------------------------------
# 3. Data loading helpers
# ---------------------------------------------------------


def load_mnist_test_dataset(data_dir: Path) -> datasets.MNIST:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # [0,1]
        ]
    )

    test_dataset = datasets.MNIST(
        root=str(data_dir),
        train=False,
        download=True,
        transform=transform,
    )
    return test_dataset


def load_single_image(image_path: Path, device: torch.device) -> torch.Tensor:
    """
    Load a single PNG/JPEG image from disk, convert to a 1x1x28x28 tensor.
    - converts to grayscale
    - resizes to 28x28
    - scales pixel values to [0,1]
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = Image.open(image_path).convert("L")  # grayscale
    img = img.resize((28, 28))

    # convert to tensor manually (0-255 -> 0-1)
    img_tensor = torch.from_numpy(
        (torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
         .view(28, 28)
         .float()
         / 255.0).numpy()
    )

    # shape: [1, 1, 28, 28]
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0).to(device)
    return img_tensor


# ---------------------------------------------------------
# 4. Inference helpers
# ---------------------------------------------------------


@torch.no_grad()
def predict_digit(model: MNISTMLP, x: torch.Tensor) -> Dict[str, Any]:
    """
    Given a single image batch (shape [1, 1, 28, 28]), return:
      - predicted digit
      - probabilities over 10 classes
    """
    logits = model(x)  # [1, 10]
    probs = torch.softmax(logits, dim=1)  # [1, 10]
    prob_values, pred_indices = torch.max(probs, dim=1)

    predicted_digit = int(pred_indices.item())
    predicted_prob = float(prob_values.item())
    probs_list = probs.squeeze(0).cpu().tolist()

    return {
        "predicted_digit": predicted_digit,
        "predicted_prob": predicted_prob,
        "probabilities": probs_list,
    }


# ---------------------------------------------------------
# 5. Main CLI logic
# ---------------------------------------------------------


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
        help="Index of the sample in the MNIST test set to visualize/predict. "
             "If provided, --image-path is ignored.",
    )
    parser.add_argument(
        "--image-path",
        type=str,
        default=None,
        help="Path to a custom image (28x28 or will be resized). "
             "If not provided, a sample from the MNIST test set is used.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    project_root = Path(__file__).resolve().parent.parent
    checkpoint_path = project_root / args.checkpoint
    data_dir = project_root / "data"

    # 1) Load model + config
    model, cfg_dict = load_checkpoint(checkpoint_path, device=device)
    print(f"[INFO] Loaded checkpoint from {checkpoint_path}")
    if cfg_dict:
        print(f"[INFO] Model config (keys): {list(cfg_dict.keys())}")

    # 2) Decide whether to use MNIST test set or custom image
    if args.image_path:
        # custom image path wins only if index is None
        if args.index is not None:
            print("[WARN] --index is ignored because --image-path was provided.")
        img_path = Path(args.image_path)
        x = load_single_image(img_path, device=device)
        true_label: Optional[int] = None
        source = f"Custom image: {img_path}"
    else:
        # use MNIST test set
        test_dataset = load_mnist_test_dataset(data_dir)
        if args.index is None:
            import random

            idx = random.randrange(len(test_dataset))
        else:
            idx = args.index
            if idx < 0 or idx >= len(test_dataset):
                raise IndexError(
                    f"Index {idx} is out of range for MNIST test set (0..{len(test_dataset)-1})."
                )

        x_img, y_label = test_dataset[idx]
        x = x_img.unsqueeze(0).to(device)  # [1, 1, 28, 28]
        true_label = int(y_label)
        source = f"MNIST test set index={idx}"

    # 3) Run prediction
    result = predict_digit(model, x)

    # 4) Print nicely
    print()
    print("[INFERENCE RE]")
