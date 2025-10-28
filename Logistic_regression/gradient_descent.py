'''
Logistic Regression using Gradient Descent
----------------------------------------
This script implements simple logistic regression from scratch
using gradient descent, without sklearn.

'''

# In Class "X" is 2D matrix while "y" is Vector which is later converted into column vector

import numpy as np

# Sigmoid Function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

class LogisticRegressionGD:
    def __init__(self):
        self.weights = None
        self.bias = None

    def iteration(self,epoches):
        if epoches < 0:
            raise ValueError("Iteration must be positive")
        self.epoches = epoches
        return self  # Required for chaining
    
    def learning_rate(self,alpha):
        if alpha <= 0:
            raise ValueError("Learning rate must be higher than 0")
        self.alpha = alpha
        return self  # Required for chaining

    def fit(self, X, y):
        n_samples = X.shape[0]
        n_features = X.shape[1]
        # Initialize weights and bias
        self.weights = np.zeros((n_features, 1))
        self.bias = 0

        # Reshape y to column vector if needed
        y = y.reshape(-1, 1)

        # Gradient descent
        for _ in range(self.epoches):
            z = np.dot(X, self.weights) + self.bias
            # Apply sigmoid to get probabilities
            y_predict = sigmoid(z)

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predict - y))
            db = (1 / n_samples) * np.sum(y_predict - y)

            # Update parameters
            self.weights -= self.alpha * dw
            self.bias -= self.alpha * db
        return self  # Required for chaining

    def predict_probab(self, X):
        z = np.dot(X, self.weights) + self.bias
        return sigmoid(z)
      
    def predict(self, X):
        y_predict = self.predict_probab(X)
        return np.where(y_predict > 0.5, 1, 0)
    

if __name__  == "__main__":
    model = LogisticRegressionGD()
    x = np.array([[1], [2], [3], [4], [5]])
    y = np.array([0, 1, 1, 0, 1])
    model.learning_rate(0.001).iteration(100000).fit(x,y)
    print(model.predict_probab(x))
    print(model.predict(x))