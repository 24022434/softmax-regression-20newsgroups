import os
import sys
# Add project root to sys.path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.softmax import SoftmaxRegression
from src.utils import load_data, vectorize, accuracy_score
import matplotlib.pyplot as plt

def evaluate(model_path='results/model.npy'):
    # Load test data
    test_texts, test_labels = load_data(subset='test')
    # Vectorize using same vectorizer saved during training (assuming saved)
    # For simplicity, re-fit vectorizer on training data (not ideal) – in practice, save vectorizer.
    # Here we just load training data to fit vectorizer.
    train_texts, _ = load_data(subset='train')
    _, X_test, vectorizer = vectorize(train_texts, test_texts, max_features=20000)
    X_test = X_test
    # Load model parameters if saved
    if os.path.exists(model_path):
        params = np.load(model_path, allow_pickle=True).item()
        model = SoftmaxRegression(n_features=params['W'].shape[0], n_classes=params['W'].shape[1])
        model.W = params['W']
        model.b = params['b']
    else:
        raise FileNotFoundError(f"Model file not found: {model_path}")
    # Predict and compute accuracy
    y_pred = model.predict(X_test)
    acc = accuracy_score(test_labels, y_pred)
    print(f"Test Accuracy: {acc:.4f}")
    return acc

if __name__ == '__main__':
    evaluate()
