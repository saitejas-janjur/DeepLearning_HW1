import numpy as np
import sys

"""This script implements a two-class logistic regression model.
"""

class logistic_regression(object):
	
    def __init__(self, learning_rate, max_iter):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.W = None

    def fit_BGD(self, X, y):
        """Train perceptron model on data (X,y) with Batch Gradient Descent.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            self: Returns an instance of self.
        """
        n_samples, n_features = X.shape

		### YOUR CODE HERE
        n_samples, n_features = X.shape
        self.W = np.zeros(n_features)
        for _ in range(self.max_iter):
            model = np.dot(X, self.W)
            predictions = 1 / (1 + np.exp(-model))
            errors = predictions - y
            gradient = np.dot(X.T, errors) / n_samples
            self.W -= self.learning_rate * gradient

		### END YOUR CODE
        return self

    def fit_miniBGD(self, X, y, batch_size):
        """Train perceptron model on data (X,y) with mini-Batch Gradient Descent.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.
            batch_size: An integer.

        Returns:
            self: Returns an instance of self.
        """
		### YOUR CODE HERE
        n_samples, n_features = X.shape
        self.W = np.zeros(n_features)
        for _ in range(self.max_iter):
            for i in range(0, n_samples, batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                model = np.dot(X_batch, self.W)
                predictions = 1 / (1 + np.exp(-model))
                errors = predictions - y_batch
                gradient = np.dot(X_batch.T, errors) / batch_size
                self.W -= self.learning_rate * gradient
		### END YOUR CODE
        return self

    def fit_SGD(self, X, y):
        """Train perceptron model on data (X,y) with Stochastic Gradient Descent.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            self: Returns an instance of self.
        """
		### YOUR CODE HERE
        n_samples, n_features = X.shape
        self.W = np.zeros(n_features)
        for _ in range(self.max_iter):
            for i in range(n_samples):
                model = np.dot(X[i], self.W)
                predictions = 1 / (1 + np.exp(-model))
                errors = predictions - y[i]
                gradient = X[i] * errors
                self.W -= self.learning_rate * gradient
		### END YOUR CODE
        return self

    def _gradient(self, _x, _y):
        """Compute the gradient of cross-entropy with respect to self.W
        for one training sample (_x, _y). This function is used in fit_*.

        Args:
            _x: An array of shape [n_features,].
            _y: An integer. 1 or -1.

        Returns:
            _g: An array of shape [n_features,]. The gradient of
                cross-entropy with respect to self.W.
        """
		### YOUR CODE HERE
        if _y == -1:
            _y = 0

        # Calculate the hypothesis (prediction) using the sigmoid function
        z = np.dot(_x, self.W)
        hypothesis = 1 / (1 + np.exp(-z))
        
        # Calculate the gradient
        _g = (_x * (hypothesis - _y))

        return _g
		### END YOUR CODE

    def get_params(self):
        """Get parameters for this perceptron model.

        Returns:
            W: An array of shape [n_features,].
        """
        if self.W is None:
            print("Run fit first!")
            sys.exit(-1)
        return self.W

    def predict_proba(self, X):
        """Predict class probabilities for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].

        Returns:
            preds_proba: An array of shape [n_samples, 2].
                Only contains floats between [0,1].
        """
		### YOUR CODE HERE
        model = np.dot(X, self.W)
        predictions = 1 / (1 + np.exp(-model))
        return predictions
		### END YOUR CODE


    def predict(self, X):
        """Predict class labels for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].

        Returns:
            preds: An array of shape [n_samples,]. Only contains 1 or -1.
        """
		### YOUR CODE HERE
        probabilities = self.predict_proba(X)
        return np.where(probabilities >= 0.5, 1, 0)
		### END YOUR CODE

    def score(self, X, y):
        """Returns the mean accuracy on the given test data and labels.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            score: An float. Mean accuracy of self.predict(X) wrt. y.
        """
		### YOUR CODE HERE
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy
		### END YOUR CODE
    
    def assign_weights(self, weights):
        self.W = weights
        return self