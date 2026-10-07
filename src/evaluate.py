"""Evaluate the trained CNN and visualise predictions."""
import os
import torch
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from src.model import ShapeCNN
from src.dataset import ShapesDataset

torch.manual_seed(0)

dataset = ShapesDataset(n_samples=3000, seed=0)
_, test_ds = random_split(dataset, [2400, 600])
test_loader = DataLoader(test_ds, batch_size=128)

model = ShapeCNN(num_classes=3)
model.load_state_dict(torch.load("saved/cnn_shapes.pt", weights_only=True))
model.eval()

total, correct = 0, 0
predictions = []
with torch.no_grad():
    for images, labels in test_loader:
        out = model(images)
        total += labels.size(0)
        correct += (out.argmax(1) == labels).sum().item()
        predictions.append((images, labels, out.argmax(1)))

print(f"Test accuracy: {correct/total:.3f} ({correct}/{total})")

images, labels, preds = predictions[0]
os.makedirs("plots", exist_ok=True)
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.flat):
    ax.imshow(images[i][0], cmap="gray")
    true_name = ShapesDataset.LABELS[labels[i]]
    pred_name = ShapesDataset.LABELS[preds[i]]
    color = "green" if true_name == pred_name else "red"
    ax.set_title(f"true: {true_name}\npred: {pred_name}", color=color, fontsize=9)
    ax.axis("off")
plt.tight_layout()
plt.savefig("plots/predictions.png", dpi=120)
print("Saved prediction grid to plots/predictions.png")
