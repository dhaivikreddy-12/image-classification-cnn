# 🖼️ Image Classification CNN

> *"MNIST is called the Hello World of deep learning — but the hello world is where you learn to debug."*

My first real deep learning project: a convolutional neural network in PyTorch that classifies **70,000 real handwritten digits** from the MNIST dataset. Nothing synthetic, nothing toy — this is the actual benchmark every CNN tutorial starts with, and it's harder than it looks.

## What this project does

- Downloads the real MNIST dataset (60,000 train + 10,000 test handwritten digits).
- Normalises pixel values and builds `DataLoader`s with an 80/10 train/val split.
- Defines a CNN with three convolution blocks, batch norm, max pooling, and dropout.
- Trains for up to 5 epochs with Adam and a step LR schedule, checkpointing the best model.
- Evaluates on the official 10,000-image test split.
- Renders a grid of predictions, green for correct and red for wrong.

## The dataset

[MNIST](http://yann.lecun.com/exdb/mnist/) — 70,000 greyscale 28×28 images of handwritten digits 0–9, from US Census clerks and high school students.

| Split | Images |
|---|---|
| Train (54,000 used) | Handwritten digits |
| Validation (6,000) | Held out from train for checkpointing |
| Test (10,000) | Official test split |

## How to run it

```bash
pip install -r requirements.txt

python -m src.train_cnn   # trains, saves best checkpoint to saved/mnist_cnn.pt
python -m src.evaluate    # scores the test split, saves plots/predictions.png
```

Runs in a few minutes on CPU. A GPU is used automatically if one is available.

## Project structure

```
image-classification-cnn/
├── src/
│   ├── dataset.py     # MNIST loader via torchvision
│   ├── model.py       # DigitCNN definition
│   ├── train_cnn.py   # training loop with checkpointing
│   └── evaluate.py    # test scoring + prediction grid
├── data/              # MNIST cache
├── saved/             # best checkpoint
├── tests/
├── requirements.txt
└── README.md
```

## What I learned

- That batch normalisation stabilises training more than I expected it to.
- Why you always need a validation split separate from test — otherwise you're selecting on your own exam.
- How much a LR schedule matters: without it the last epochs overfit fast.
- That 99%+ on MNIST is achievable, and that this is precisely why MNIST is now considered *too easy* — real work needs CIFAR or beyond.

## Results

| Split | Accuracy |
|---|---|
| Validation | 0.9942 |
| **Test (10,000 images)** | **0.9938** (9,938 / 10,000) |

Training took ~5 minutes on CPU across 5 epochs. The prediction grid in `plots/predictions.png` shows where the model actually struggles — mostly ambiguous handwriting rather than obvious errors.

---

*Built with Python, PyTorch, torchvision, matplotlib. Real image data, honestly measured.*
