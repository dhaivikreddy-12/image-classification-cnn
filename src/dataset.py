"""Synthetic shape dataset (circle, square, triangle)."""
import numpy as np
import torch
from torch.utils.data import Dataset


def _draw_shape(kind, size=32, angle_deg=0.0):
    img = np.zeros((size, size), dtype=np.float32)
    yy, xx = np.mgrid[0:size, 0:size]
    cx, cy = size / 2, size / 2
    rad = np.deg2rad(angle_deg)
    xr = (xx - cx) * np.cos(rad) - (yy - cy) * np.sin(rad) + cx
    yr = (xx - cx) * np.sin(rad) + (yy - cy) * np.cos(rad) + cy
    if kind == 0:  # circle
        r = 9
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
        img[mask] = 1.0
    elif kind == 1:  # square
        half = 9
        mask = (np.abs(xr - cx) <= half) & (np.abs(yr - cy) <= half)
        img[mask] = 1.0
    else:  # triangle
        pts = np.array([[cx, cy - 10], [cx - 10, cy + 8], [cx + 10, cy + 8]], dtype=np.float32)
        pts_r = np.zeros_like(pts)
        for i in range(3):
            px, py = pts[i]
            pts_r[i] = [(px - cx) * np.cos(rad) - (py - cy) * np.sin(rad) + cx,
                        (px - cx) * np.sin(rad) + (py - cy) * np.cos(rad) + cy]
        x, y = xx.flatten(), yy.flatten()
        inside = np.zeros_like(x, dtype=bool)
        for i in range(3):
            x1, y1 = pts_r[i]
            x2, y2 = pts_r[(i + 1) % 3]
            inside |= ((y2 - y1) * (x - x1) - (x2 - x1) * (y - y1)) >= 0
        img[inside.reshape(size, size)] = 1.0
    return img


class ShapesDataset(Dataset):
    LABELS = ["circle", "square", "triangle"]

    def __init__(self, n_samples=3000, size=32, noise=0.02, seed=0):
        rng = np.random.default_rng(seed)
        self.images, self.labels = [], []
        for _ in range(n_samples):
            kind = int(rng.integers(0, 3))
            angle = rng.uniform(-40, 40)
            img = _draw_shape(kind, size, angle_deg=angle)
            img += rng.normal(0, noise, img.shape)
            img = np.clip(img, 0, 1)
            self.images.append(img[None, :, :])
            self.labels.append(kind)
        self.images = torch.tensor(np.stack(self.images), dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


if __name__ == "__main__":
    ds = ShapesDataset(n_samples=5)
    print(f"Dataset has {len(ds)} samples, image shape {ds[0][0].shape}")
    print(f"Labels: {ds.LABELS}")
