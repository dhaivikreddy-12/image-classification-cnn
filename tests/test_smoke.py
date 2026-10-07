"""Real tests: exercise the model forward pass and the pipeline, not just file presence."""
import pathlib
import sys

import numpy as np
import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_readme_and_license_present():
    assert (ROOT / "README.md").is_file()
    assert (ROOT / "LICENSE").is_file()


def test_cnn_outputs_one_logit_per_class():
    from src.model import DigitCNN

    model = DigitCNN(num_classes=10)
    batch = torch.zeros(4, 1, 28, 28)
    out = model(batch)
    assert out.shape == (4, 10), "expected one logit per digit class"


def test_cnn_backward_pass_runs():
    """A real gradient check - catches broken layer wiring."""
    from src.model import DigitCNN

    model = DigitCNN(num_classes=10)
    out = model(torch.randn(2, 1, 28, 28))
    out.sum().backward()
    grad = model.features[0].weight.grad
    assert grad is not None, "no gradient reached the first conv layer"
    assert torch.isfinite(grad).all(), "gradients contain NaN/Inf"


def test_cnn_uses_batchnorm_and_dropout():
    from src.model import DigitCNN

    model = DigitCNN()
    assert any(isinstance(m, torch.nn.BatchNorm2d) for m in model.modules())
    assert any(isinstance(m, torch.nn.Dropout) for m in model.modules())


def test_mnist_loader_shape_is_28x28():
    """Cheap shape assertion - skipped unless data is already cached."""
    import os

    cache = ROOT / "data"
    if not cache.exists() or not any(cache.glob("**")):
        pytest.skip("MNIST not downloaded yet; run 'python -m src.train_cnn' first")

    from src.dataset import load_mnist

    img, label = load_mnist(train=False)[0]
    assert tuple(img.shape) == (1, 28, 28)
    assert 0 <= int(label) <= 9
