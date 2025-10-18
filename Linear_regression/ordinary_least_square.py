"""
Linear Regression using Ordinary Least Squares (OLS)
----------------------------------------------------
This script implements simple linear regression from scratch
using the closed-form (OLS) solution without any ML libraries.

"""

import numpy as np

# Sample data
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

# Calculate slope (m) and intercept (c)
m = np.sum((x - np.mean(x)) * (y - np.mean(y))) / np.sum((x - np.mean(x)) ** 2)
c = np.mean(y) - m * np.mean(x)

print(f"Slope (m): {m:.4f}")
print(f"Intercept (c): {c:.4f}")

# Prediction function
def predict(new_x):
    return m * new_x + c

# Example prediction
print("Predicted value for x=6:", predict(6))