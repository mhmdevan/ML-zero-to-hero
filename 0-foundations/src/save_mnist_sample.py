# save_mnist_sample.py
import os
from pathlib import Path

from torchvision import datasets, transforms, utils


def resolve_paths() -> tuple[Path, Path]:
    project_root = Path(__file__).resolve().parent

    data_dir_raw = os.getenv("FOUNDATIONS_MNIST_DATA_DIR")
    output_path_raw = os.getenv("FOUNDATIONS_MNIST_SAMPLE_OUTPUT")

    data_dir = Path(data_dir_raw) if data_dir_raw else (project_root / "data")
    output_path = Path(output_path_raw) if output_path_raw else (project_root / "digit_42.png")
    return data_dir, output_path


def main() -> None:
    data_dir, output_path = resolve_paths()

    transform = transforms.ToTensor()

    test_dataset = datasets.MNIST(
        root=str(data_dir),
        train=False,
        download=True,
        transform=transform,
    )

    index = 42
    img_tensor, label = test_dataset[index]  # img: [1, 28, 28]

    print(f"[INFO] Selected sample index={index}, true label={label}")

    utils.save_image(img_tensor, output_path)

    print(f"[INFO] Saved MNIST sample to {output_path}")


if __name__ == "__main__":
    main()
