import os
import sys
# Add project root to sys.path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.softmax import SoftmaxRegression
from src.utils import load_data, vectorize, batch_generator, accuracy_score
import matplotlib.pyplot as plt

def main():
    # Load raw data
    train_texts, train_labels = load_data(subset='train')
    test_texts, test_labels = load_data(subset='test')

    # Vectorize texts
    X_train, X_test, _ = vectorize(train_texts, test_texts, max_features=20000)
    n_features = X_train.shape[1]
    n_classes = np.max(train_labels) + 1

    # Initialize model
    model = SoftmaxRegression(n_features=n_features, n_classes=n_classes, learning_rate=0.05, reg_lambda=0.001)

    # Training parameters
    epochs = 100
    batch_size = 128

    # Train model
    loss_history = model.fit(X_train, train_labels, epochs=epochs, batch_size=batch_size, verbose=True)

    # Save loss curve
    os.makedirs('results', exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o')
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(os.path.join('results', 'loss_curve.png'))
    plt.close()
    # Save model parameters
    model_path = os.path.join('results', 'model.npy')
    np.save(model_path, {'W': model.W, 'b': model.b})

    # Evaluate on test set
    y_pred = model.predict(X_test)
    acc = accuracy_score(test_labels, y_pred)
    with open(os.path.join('results', 'accuracy.txt'), 'w') as f:
        f.write(f'Test Accuracy: {acc:.4f}\n')
    print(f'Test Accuracy: {acc:.4f}')

if __name__ == '__main__':
    main()
