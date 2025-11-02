"""
Linear Regression (Ordinary Least Squares) for Multiple Feature
------------------------------------------
Implements simple linear regression from scratch using OLS method.

"""

import numpy as np

class Multiple_OLS:

    def __init__ (self, weights):
        # Initialize weights
        self.weights = None

    def fit(self, X, y):
        ones = np.ones((len(X), 1))
        X = np.c_[ones, X]

        # Normal equation parts
        XTX = X.T @ X
        XTy = X.T @ y

        self.weights = np.linalg.inv(XTX) @ XTy  # Where W = (X^T * X)^-1 * X^T * y

        print("Training complete. Coefficients:", self.weights)

    def predict(self, X):
        if self.weights is None:
            raise ValueError("Model is not trained yet. Call fit(X, y) first.")

        ones = np.ones((len(X), 1))
        X = np.c_[ones, X]
        return X @ self.weights
    

# Testing
if __name__ == "__main__":
    X = np.array([[1, 2], [2, 3], [3, 4], [5, 6]])
    y = np.array([5, 6, 7, 9])

    model = Multiple_OLS()
    model.fit(X, y)

    test = np.array([[4, 5]])
    predict = model.predict(test)
    print(f"Predicted: {predict}")
