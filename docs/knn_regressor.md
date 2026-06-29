# K-Nearest Neighbors (KNN) Regressor

## Overview

K-Nearest Neighbors (KNN) Regressor is a supervised machine learning algorithm used for regression tasks. Unlike KNN Classifier, which predicts the most common class among the nearest neighbors, KNN Regressor predicts a continuous value by averaging the target values of the nearest neighbors.

This implementation uses the Euclidean distance metric and simple averaging for prediction.

---

# How It Works

The algorithm follows these steps:

1. Store the training dataset.
2. Compute the distance between the query sample and every training sample.
3. Select the K nearest neighbors.
4. Compute the average of their target values.
5. Return the average as the prediction.

---

# Euclidean Distance

The distance between two samples is computed using:

```text
Distance = √Σ(xᵢ − yᵢ)²
```

Samples with smaller distances are considered more similar.

---

# Prediction

Suppose the K nearest neighbors have target values:

```text
10
12
14
```

Prediction:

```text
(10 + 12 + 14) / 3 = 12
```

The predicted value is simply the arithmetic mean of the nearest neighbors.

---

# Choosing K

The value of **K** controls the smoothness of the predictions.

Small K:

- More flexible
- Captures local patterns
- More sensitive to noise

Large K:

- Smoother predictions
- Less sensitive to noise
- May underfit the data

Choosing an appropriate K is important for balancing bias and variance.

---

# Training

KNN Regressor does not learn model parameters.

Training simply stores:

- Feature matrix
- Target values

---

# Prediction Workflow

For every query sample:

1. Compute distances to all training samples.
2. Sort distances.
3. Select the K nearest neighbors.
4. Compute the average of their target values.
5. Return the average.

---

# Implementation Details

Current implementation includes:

- Euclidean distance metric
- Configurable number of neighbors (k)
- Mean-based prediction
- Continuous target prediction

---

# Computational Complexity

| Operation | Complexity |
|----------|------------|
| Fit | O(1) *(stores training data)* |
| Predict | O(n_samples × n_features) per query |

---

# Example

```python
from mini_sklearn.neighbors import KNeighborsRegressor

model = KNeighborsRegressor(n_neighbors=3)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
```

---

# Comparison with scikit-learn

| Feature | This Project | scikit-learn |
|----------|--------------|--------------|
| Regression | ✅ | ✅ |
| Euclidean Distance | ✅ | ✅ |
| Mean Prediction | ✅ | ✅ |
| Distance Weighting | ❌ | ✅ |
| Multiple Distance Metrics | ❌ | ✅ |
| KD-Tree | ❌ | ✅ |
| Ball Tree | ❌ | ✅ |

---

# Future Improvements

- Distance-weighted regression
- Manhattan distance
- Minkowski distance
- Cosine distance
- KD-Tree optimization
- Ball Tree optimization
- Radius Neighbors Regressor