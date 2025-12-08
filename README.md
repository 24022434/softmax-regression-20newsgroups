# Softmax Regression on 20 Newsgroups

This repository implements a multi‑class Softmax Regression model from scratch, trained with mini‑batch gradient descent, to classify text documents from the **20 Newsgroups** dataset.

## Project Structure
```
softmax-regression-20newsgroups/
│   README.md
│   requirements.txt
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

## Setup
```bash
# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Run Training
```bash
python src/train.py
```
This will:
1. Load the 20 Newsgroups data.
2. Vectorize the text using TF‑IDF (20 000 features).
3. Train the Softmax Regression model.
4. Save a loss curve plot to `results/loss_curve.png`.
5. Write the test accuracy to `results/accuracy.txt`.

## Evaluate Separately
```bash
python src/evaluation.py
```
The script loads the saved model parameters (if you modify `train.py` to save them) and prints the test accuracy.

## Notebook Demo
Open the notebook to step through the pipeline interactively:
```bash
jupyter notebook notebooks/demo_training.ipynb
```

## Notes
- The dataset is fetched directly from `sklearn.datasets.fetch_20newsgroups`; no data files are stored in the repository.
- Hyper‑parameters such as learning rate, batch size, and number of epochs can be adjusted in `src/train.py`.
- For reproducibility, you may set a random seed in the training script.