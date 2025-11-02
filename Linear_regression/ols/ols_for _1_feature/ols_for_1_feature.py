"""
Linear Regression (Ordinary Least Squares) for 1 Feature
------------------------------------------
Implements simple linear regression from scratch using OLS method.

"""

import numpy as np


class Single_OLS:
    def __init__(self):
        # Initialize slope and intercept.
        self.m = None
        self.c = None

    def fit(self, x, y):
        # Fit the model to the data using the OLS formula.
        x = np.array(x).flatten()
        y = np.array(y).flatten()

        # Calculate means of x and y    
        x_mean = np.mean(x)
        y_mean = np.mean(y)

        num = np.sum((x - x_mean) * (y - y_mean))    # numerator
        den = np.sum((x - x_mean)**2)                # denominator

        # Calculate slope (m) and intercept (c)
        self.m = num / den
        # c = mean(y) - m * mean(x)
        self.c = y_mean - self.m * x_mean

    def predict(self, x):
        # Predict y values for given x.
        if self.m is None or self.c is None:
            raise ValueError("Call fit(x, y) first.")
        return self.m * np.array(x).flatten() + self.c

    def coefficients(self):
        # Return the slope and intercept.
        return self.m, self.c


# Testing
if __name__ == "__main__":
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 5, 4, 5]

    model = Single_OLS()
    model.fit(x, y)

    print(model)
    print("Predicted value for x=6:", model.predict(6))
