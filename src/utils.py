import os
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

def load_data(subset='train'):
    """Load 20 Newsgroups data for the given subset.
    Returns a list of raw text documents and their target labels.
    """
    data = fetch_20newsgroups(subset=subset, remove=('headers', 'footers', 'quotes'))
    return data.data, data.target

def vectorize(train_texts, test_texts, max_features=20000):
    """Fit a TF‑IDF vectorizer on training texts and transform both train and test.
    Returns (X_train, X_test, vectorizer).
    """
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    return X_train.toarray(), X_test.toarray(), vectorizer

def batch_generator(X, y, batch_size=64, shuffle=True):
    """Yield batches of (X_batch, y_batch).
    y should be a 1‑D array of integer class labels.
    """
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    if shuffle:
        np.random.shuffle(indices)
    for start in range(0, n_samples, batch_size):
        end = start + batch_size
        batch_idx = indices[start:end]
        yield X[batch_idx], y[batch_idx]

def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)
