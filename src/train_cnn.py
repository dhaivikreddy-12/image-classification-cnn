"""Train the CNN."""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from src.model import ShapeCNN
from src.dataset import ShapesDataset

torch.manual_seed(0)

dataset = ShapesDataset(n_samples=3000, seed=0)
train_ds, val_ds = random_split(dataset, [2400, 600])
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=128)

model = ShapeCNN(num_classes=3)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

os.makedirs("saved", exist_ok=True)
epochs = 8
best_acc = 0.0

for epoch in range(1, epochs + 1):
    model.train()
    total, correct, running_loss = 0, 0, 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        out = model(images)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * len(images)
        total += labels.size(0)
        correct += (out.argmax(1) == labels).sum().item()
    train_acc = correct / total

    model.eval()
    v_total, v_correct = 0, 0
    with torch.no_grad():
        for images, labels in val_loader if False else val_loader:
            out = model(images)
            v_total += labels.size(0)
            v_correct += (out.argmax(1) == labels).sum().item()
    val_acc = v_correct / v_total

    print(f"Epoch {epoch:>2}/{epochs} | loss {running_loss/total:.4f} "
          f"| train acc {train_acc:.3f} | val acc {val_acc:.3f}")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "saved/cnn_shapes.pt")
        print(f"  -> saved best model (val acc {val_acc:.3f})")

print(f"\nBest validation accuracy: {best_acc:.3f}")
