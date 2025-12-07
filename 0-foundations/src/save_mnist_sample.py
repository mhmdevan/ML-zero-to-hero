# save_mnist_sample.py
from pathlib import Path

import torch
from torchvision import datasets, transforms, utils


def main() -> None:
    project_root = Path(__file__).resolve().parent
    data_dir = project_root / "data"
    output_path = project_root / "digit_42.png" 

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
