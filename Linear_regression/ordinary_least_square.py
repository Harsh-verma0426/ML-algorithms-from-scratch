"""
Linear Regression (Ordinary Least Squares)
------------------------------------------
Implements simple linear regression from scratch using OLS method.

"""

import numpy as np


class LinearRegressionOLS:
    def __init__(self):
        # Initialize slope and intercept.
        self.m = None
        self.c = None

    def fit(self, x, y):
        # Fit the model to the data using the OLS formula.
        x = np.array(x)
        y = np.array(y)

        # Calculate slope (m) and intercept (c)
        self.m = np.sum((x - np.mean(x)) * (y - np.mean(y))) / np.sum((x - np.mean(x)) ** 2)
        self.c = np.mean(y) - self.m * np.mean(x)

    def predict(self, x):
        # Predict y values for given x.
        if self.m is None or self.c is None:
            raise ValueError("Call fit(x, y) first.")
        return self.m * np.array(x) + self.c

    def coefficients(self):
        # Return the slope and intercept.
        return self.m, self.c


if __name__ == "__main__":
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 5, 4, 5]

    model = LinearRegressionOLS()
    model.fit(x, y)

    print(model)
    print("Predicted value for x=6:", model.predict(6))
