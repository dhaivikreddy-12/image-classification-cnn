"""Train a CNN on the real MNIST handwritten digits dataset."""
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from src.model import DigitCNN
from src.dataset import load_mnist

torch.manual_seed(0)

SAVE_DIR = "saved"
EPOCHS = 5
BATCH_SIZE = 128

full_train = load_mnist(train=True)
test_ds = load_mnist(train=False)

# Hold out a validation slice from the official training split.
train_size = int(0.9 * len(full_train))
val_size = len(full_train) - train_size
train_ds, val_ds = random_split(full_train, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_ds, batch_size=512)

print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model = DigitCNN(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

os.makedirs(SAVE_DIR, exist_ok=True)
best_val = 0.0

for epoch in range(1, EPOCHS + 1):
    start = time.time()
    model.train()
    total = correct = 0
    running = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(images)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        running += loss.item() * images.size(0)
        total += labels.size(0)
        correct += (out.argmax(1) == labels).sum().item()
    scheduler.step()
    train_acc = correct / total

    model.eval()
    v_total = v_correct = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            out = model(images)
            v_total += labels.size(0)
            v_correct += (out.argmax(1) == labels).sum().item()
    val_acc = v_correct / v_total

    print(
        f"Epoch {epoch}/{EPOCHS} | loss {running/total:.4f} "
        f"| train {train_acc:.4f} | val {val_acc:.4f} | {time.time()-start:.1f}s"
    )

    if val_acc > best_val:
        best_val = val_acc
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, "mnist_cnn.pt"))
        print(f"  saved new best (val {val_acc:.4f})")

print(f"\nBest validation accuracy: {best_val:.4f}")
