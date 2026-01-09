import os
import sys
# Add project root to sys.path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.utils import load_data, vectorize
from sklearn.model_selection import train_test_split

def get_dataset(max_features=20000, val_ratio=0.1, random_state=36):
    """Load 20 Newsgroups data, vectorize with TF‑IDF, and split into train/val/test.

    Returns:
        X_train, X_val, X_test: np.ndarray of shape (n_samples, n_features)
        y_train, y_val, y_test: np.ndarray of integer class labels
    """
    # Load raw texts
    train_texts, train_labels = load_data(subset='train')
    test_texts, test_labels = load_data(subset='test')

    # Vectorize using training data to fit the TF‑IDF
    X_train_full, X_test, vectorizer = vectorize(train_texts, test_texts, max_features=max_features)
    y_train_full = np.array(train_labels)
    y_test = np.array(test_labels)

    # Split a validation set from the training data
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=val_ratio, random_state=random_state, stratify=y_train_full
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
