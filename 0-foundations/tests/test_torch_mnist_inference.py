from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def test_predict_digit_returns_valid_distribution() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    module = pytest.importorskip("src.torch_mnist_inference")

    model = module.MNISTMLP(hidden_dims=(32, 16), num_classes=10)
    x = torch.randn(1, 1, 28, 28)

    result = module.predict_digit(model, x)

    probs = np.asarray(result["probabilities"], dtype=float)
    assert result["predicted_digit"] in range(10)
    assert probs.shape == (10,)
    assert np.isclose(float(np.sum(probs)), 1.0, atol=1e-6)


def test_load_single_image_returns_expected_shape(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    module = pytest.importorskip("src.torch_mnist_inference")

    img = (np.ones((28, 28), dtype=np.uint8) * 127)
    image_path = Path(tmp_path) / "digit.png"
    Image.fromarray(img, mode="L").save(image_path)

    tensor = module.load_single_image(image_path, device=torch.device("cpu"))
    assert tensor.shape == (1, 1, 28, 28)
