"""MNIST dataset loader using torchvision, with a local cache."""
import os
from torchvision import datasets, transforms

DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

CLASSES = [str(i) for i in range(10)]


def load_mnist(train=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    return datasets.MNIST(root=DATA_ROOT, train=train, download=True, transform=transform)


if __name__ == "__main__":
    ds = load_mnist(train=True)
    print(f"MNIST train split: {len(ds)} images, classes: {CLASSES}")
    x, y = ds[0]
    print(f"Sample image shape: {tuple(x.shape)}, label: {y}")
