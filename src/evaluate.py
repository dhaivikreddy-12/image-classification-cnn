"""Evaluate the trained CNN on the MNIST test split and visualise predictions."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset

from src.model import DigitCNN
from src.dataset import load_mnist

torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_ds = load_mnist(train=False)
test_loader = DataLoader(test_ds, batch_size=512)

model = DigitCNN(num_classes=10).to(device)
model.load_state_dict(torch.load(os.path.join("saved", "mnist_cnn.pt"), weights_only=True))
model.eval()

total = correct = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        out = model(images)
        total += labels.size(0)
        correct += (out.argmax(1) == labels).sum().item()

print(f"Test accuracy: {correct/total:.4f} ({correct}/{total})")

# Visualise the first 20 test images with predictions.
subset = Subset(test_ds, range(20))
os.makedirs("plots", exist_ok=True)
fig, axes = plt.subplots(4, 5, figsize=(10, 8))
with torch.no_grad():
    for ax, idx in zip(axes.flat, range(20)):
        img, label = subset[idx]
        pred = model(img.unsqueeze(0).to(device)).argmax(1).item()
        ax.imshow(img.squeeze(0), cmap="gray")
        color = "green" if pred == label else "red"
        ax.set_title(f"true {label} / pred {pred}", color=color, fontsize=9)
        ax.axis("off")
plt.suptitle("MNIST test predictions", fontsize=12)
plt.tight_layout()
plt.savefig("plots/predictions.png", dpi=120)
print("Saved prediction grid to plots/predictions.png")
