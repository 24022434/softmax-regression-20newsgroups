# Softmax Regression on 20 Newsgroups

This repository implements a multi‑class Softmax Regression model from scratch, trained with mini‑batch gradient descent, to classify text documents from the **20 Newsgroups** dataset.

## Project Structure
```
softmax-regression-20newsgroups/
│   README.md
│
├── data/                # (dataset is loaded on‑the‑fly, no files stored)
├── src/
│   ├── softmax.py       # SoftmaxRegression class implementation
│   ├── utils.py         # Data loading, TF‑IDF vectorization, batching, accuracy
│   ├── dataset.py       # Helper to get train/val/test splits
│   ├── train.py         # Training script, saves loss curve & accuracy
│   └── evaluation.py    # Simple evaluation script (loads model & computes test accuracy)
│
├── notebooks/
│   └── demo_training.ipynb  # Jupyter notebook demo of the full pipeline
│
└── results/
    ├── loss_curve.png   # Generated after training
    └── accuracy.txt     # Test accuracy report
```

