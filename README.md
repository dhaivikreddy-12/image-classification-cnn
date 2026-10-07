# 🖼️ Image Classification with CNN

> *"A model that can tell a square from a circle is the same model that can tell a dog from a cat — just with more layers and data."*

My first real deep learning project. It trains a Convolutional Neural Network (CNN) in PyTorch to classify synthetic images of shapes (circle, square, triangle). I deliberately used a self-contained dataset so you can run the whole thing offline without downloading gigabytes.

## What this project does

- Generates a small dataset of synthetic shape images on the fly (no download needed).
- Builds a simple CNN with convolutional + pooling + dense layers.
- Trains it in PyTorch with a real training loop.
- Tracks train/val accuracy and loss, and saves the best model.
- Evaluates on held-out images and visualises some predictions.

## Why a synthetic dataset?

Real image datasets (CIFAR, ImageNet) are huge and slow to download. Synthetic shapes let you learn the *entire* CNN pipeline — architecture, training loop, evaluation — in minutes on any machine, CPU included. The skills transfer directly to real datasets.

## How to run it

```bash
pip install -r requirements.txt

# Train the CNN
python train_cnn.py

# Evaluate and visualise predictions
python evaluate.py
```

## Project structure

```
image-classification-cnn/
├── src/
│   ├── dataset.py          # synthetic shape dataset
│   ├── model.py            # CNN definition
│   ├── train_cnn.py        # training loop
│   └── evaluate.py         # evaluation + visualisation
├── saved/
│   └── cnn_shapes.pt       # best model checkpoint
├── requirements.txt
└── README.md
```

## What I learned

- How convolutions actually "see" local patterns in images.
- The anatomy of a training loop: forward, loss, backward, step.
- Why pooling shrinks the image while keeping important features.
- How to interpret loss curves — and why val loss bouncing around is normal.

## Results

The CNN reaches **~99% accuracy** on the shape test set within a few epochs — shapes are easy. The real win is understanding *how* it learns, which carries over to harder problems like CIFAR or real photos.

---

*Built with Python, PyTorch, torchvision. Made for learning, by a student, for students.*
