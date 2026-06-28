# Linear Regression

## Overview

Linear Regression is a supervised machine learning algorithm used to model the relationship between one or more input features and a continuous target variable.

This implementation provides two optimization methods:

* **Ordinary Least Squares (OLS)** – Computes the optimal coefficients analytically using the Normal Equation.
* **Gradient Descent (GD)** – Iteratively updates the model parameters to minimize the Mean Squared Error (MSE) loss.

The implementation follows an API similar to scikit-learn.

---

# Mathematical Model

The prediction for each sample is computed as

```text
ŷ = Xw + b
```

where

* **X** : Feature matrix
* **w** : Coefficient vector
* **b** : Bias (intercept)
* **ŷ** : Predicted output

---

# Loss Function

Training minimizes the Mean Squared Error (MSE):

```text
MSE = (1/n) Σ (y - ŷ)²
```

Lower MSE indicates a better fit to the training data.

---

# Optimization Methods

## Ordinary Least Squares (OLS)

OLS directly computes the optimal coefficients using the Normal Equation.

### Advantages

* Exact analytical solution
* No learning rate required
* Fast for small datasets

### Limitations

* Matrix inversion is computationally expensive
* Can become unstable for highly correlated features
* Less practical for very large datasets

---

## Gradient Descent

Gradient Descent updates the coefficients iteratively.

Each iteration performs:

1. Compute predictions
2. Compute gradients
3. Update coefficients
4. Repeat until convergence

### Advantages

* Works well for large datasets
* Scales better than OLS
* Forms the basis of many modern optimization algorithms

### Limitations

* Requires tuning the learning rate
* May require many iterations
* Convergence depends on feature scaling and hyperparameters

---

# Implementation Details

Current implementation includes:

* Multiple input features
* Configurable learning rate
* Configurable number of iterations
* OLS and Gradient Descent optimization
* Prediction using learned coefficients
* R² Score for evaluation

---

# Computational Complexity

| Operation              | Complexity                         |
| ---------------------- | ---------------------------------- |
| Fit (OLS)              | O(n³) (matrix inversion)           |
| Fit (Gradient Descent) | O(iterations × samples × features) |
| Predict                | O(samples × features)              |

---

# Example

```python
from mini_sklearn.linear_model import LinearRegression

model = LinearRegression(
    method="gd",
    learning_rate=0.01,
    iter=1000
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

score = model.score(y_test, predictions)
```

---

# Comparison with scikit-learn

| Feature               | This Project | scikit-learn                               |
| --------------------- | ------------ | ------------------------------------------ |
| OLS                   | ✅            | ✅                                          |
| Gradient Descent      | ✅            | ❌ (`SGDRegressor` is a separate estimator) |
| Multi-feature Support | ✅            | ✅                                          |
| Prediction            | ✅            | ✅                                          |
| R² Score              | ✅            | ✅                                          |

---

# Future Improvements

* Feature normalization utilities
* Regularized regression (Ridge, Lasso, Elastic Net)
* Early stopping
* Mini-batch Gradient Descent
* Stochastic Gradient Descent
* Sample weighting
* Numerical stability improvements