# Logistic Regression

## Overview

Logistic Regression is a supervised machine learning algorithm used for binary classification. Instead of predicting a continuous value, it estimates the probability that a sample belongs to a particular class.

Although its name contains "Regression", Logistic Regression is a classification algorithm.

This implementation trains the model using Gradient Descent and predicts class labels by applying a threshold to the predicted probabilities.

---

# Mathematical Model

The linear model is

\[
z = Xw + b
\]

where

- **X** : Feature matrix
- **w** : Coefficient vector
- **b** : Bias (intercept)

Instead of using the linear output directly, Logistic Regression passes it through the Sigmoid function.

---

# Sigmoid Function

The sigmoid function maps any real-valued input into the range (0, 1).

\[
\sigma(z)=\frac{1}{1+e^{-z}}
\]

The output represents the predicted probability of the positive class.

---

# Decision Rule

Predicted probabilities are converted into class labels using a threshold.

```
if probability >= 0.5:
    predict 1
else:
    predict 0
```

The threshold can be adjusted depending on the application.

---

# Loss Function

This implementation minimizes Binary Cross-Entropy Loss.

\[
L = -\frac{1}{n}\sum_{i=1}^{n}
\left(
y_i\log(\hat y_i)
+
(1-y_i)\log(1-\hat y_i)
\right)
\]

Gradient Descent iteratively updates the model parameters to minimize this loss.

---

# Optimization

Training consists of the following steps:

1. Compute the linear output
2. Apply the sigmoid function
3. Compute Binary Cross-Entropy Loss
4. Compute gradients
5. Update coefficients
6. Repeat for the specified number of iterations

---

# Implementation Details

Current implementation includes:

- Binary classification
- Gradient Descent optimization
- Configurable learning rate
- Configurable number of iterations
- Sigmoid activation
- Probability prediction
- Class prediction using a configurable threshold (default: 0.5)

---

# Computational Complexity

| Operation | Complexity |
|-----------|------------|
| Fit | O(iterations × samples × features) |
| Predict | O(samples × features) |

---

# Example

```python
from mini_sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    learning_rate=0.01,
    iter=1000
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
```

---

# Comparison with scikit-learn

| Feature | This Project | scikit-learn |
|----------|--------------|--------------|
| Binary Classification | ✅ | ✅ |
| Gradient Descent | ✅ | Different Solver |
| Probability Prediction | ✅ | ✅ |
| Class Prediction | ✅ | ✅ |

---

# Future Improvements

- Multi-class classification
- Multiple optimization solvers
- L1 Regularization
- L2 Regularization
- Elastic Net Regularization
- Early stopping
- Sample weighting
- Numerical stability improvements