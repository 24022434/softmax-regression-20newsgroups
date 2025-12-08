import numpy as np

class SoftmaxRegression:
    def __init__(self, n_features, n_classes, learning_rate=0.01, reg_lambda=0.0):
        self.n_features = n_features
        self.n_classes = n_classes
        self.lr = learning_rate
        self.reg_lambda = reg_lambda
        # Initialize weights with small random values
        self.W = np.random.randn(n_features, n_classes) * 0.01
        self.b = np.zeros((1, n_classes))

    def softmax(self, logits):
        # Numerically stable softmax
        exp_shift = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_shift / np.sum(exp_shift, axis=1, keepdims=True)
        return probs

    def predict_proba(self, X):
        logits = X @ self.W + self.b
        return self.softmax(logits)

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def cross_entropy_loss(self, X, y_onehot):
        m = X.shape[0]
        probs = self.predict_proba(X)
        # Clip to avoid log(0)
        eps = 1e-15
        log_probs = np.log(np.clip(probs, eps, 1 - eps))
        loss = -np.sum(y_onehot * log_probs) / m
        # L2 regularization
        if self.reg_lambda > 0:
            loss += (self.reg_lambda / (2 * m)) * np.sum(self.W * self.W)
        return loss

    def gradient(self, X, y_onehot):
        m = X.shape[0]
        probs = self.predict_proba(X)
        dZ = probs - y_onehot  # shape (m, n_classes)
        dW = (X.T @ dZ) / m
        db = np.sum(dZ, axis=0, keepdims=True) / m
        if self.reg_lambda > 0:
            dW += (self.reg_lambda / m) * self.W
        return dW, db

    def fit(self, X, y, epochs=100, batch_size=64, verbose=True):
        n_samples = X.shape[0]
        n_classes = self.n_classes
        # Convert labels to one-hot
        y_onehot = np.zeros((n_samples, n_classes))
        y_onehot[np.arange(n_samples), y] = 1
        loss_history = []
        for epoch in range(1, epochs + 1):
            # Shuffle data each epoch
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_onehot_shuffled = y_onehot[indices]
            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_onehot_shuffled[start:end]
                dW, db = self.gradient(X_batch, y_batch)
                # Parameter update
                self.W -= self.lr * dW
                self.b -= self.lr * db
            # Compute loss after epoch
            loss = self.cross_entropy_loss(X, y_onehot)
            loss_history.append(loss)
            if verbose and (epoch % 10 == 0 or epoch == 1):
                print(f"Epoch {epoch}/{epochs} - Loss: {loss:.4f}")
        return loss_history
