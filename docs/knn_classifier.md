# K-Nearest Neighbors (KNN) Classifier

## Overview

K-Nearest Neighbors (KNN) is a supervised machine learning algorithm used for classification. Unlike many machine learning algorithms, KNN does not learn an explicit mathematical model during training. Instead, it stores the training data and classifies new samples based on the labels of their nearest neighbors.

This implementation uses the Euclidean distance metric and majority voting for classification.

---

# How It Works

The algorithm follows these steps:

1. Store the training dataset.
2. Compute the distance between the query sample and every training sample.
3. Select the K nearest neighbors.
4. Count the frequency of each class among the neighbors.
5. Return the class with the highest frequency.

---

# Euclidean Distance

The distance between two samples is computed using:

```text
Distance = √Σ(xᵢ − yᵢ)²
```

Smaller distance indicates greater similarity.

---

# Majority Voting

After selecting the K nearest neighbors:

```text
Class 0 → 2 neighbors

Class 1 → 3 neighbors
```

Prediction:

```text
Class 1
```

The class with the highest number of votes is selected.

---

# Choosing K

The value of **K** controls the behavior of the classifier.

Small K:

- Sensitive to noise
- More flexible
- Higher variance

Large K:

- Smoother decision boundary
- Less sensitive to noise
- Higher bias

Choosing an appropriate K is important for good performance.

---

# Training

Unlike many machine learning algorithms, KNN does not perform optimization during training.

Training simply stores:

- Feature matrix
- Target labels

---

# Prediction

For every query sample:

1. Compute distances to all training samples.
2. Sort distances.
3. Select the K nearest neighbors.
4. Perform majority voting.
5. Return the predicted class.

---

# Implementation Details

Current implementation includes:

- Euclidean distance metric
- Configurable number of neighbors (k)
- Majority voting classification
- Binary and multiclass classification support

---

# Computational Complexity

| Operation | Complexity |
|----------|------------|
| Fit | O(1) *(stores training data)* |
| Predict | O(n_samples × n_features) per query |

---

# Example

```python
from mini_sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=3)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
```

---

# Comparison with scikit-learn

| Feature | This Project | scikit-learn |
|----------|--------------|--------------|
| Classification | ✅ | ✅ |
| Euclidean Distance | ✅ | ✅ |
| Binary Classification | ✅ | ✅ |
| Multiclass Classification | ✅ | ✅ |
| Distance Weighting | ❌ | ✅ |
| Multiple Distance Metrics | ❌ | ✅ |
| KD-Tree | ❌ | ✅ |
| Ball Tree | ❌ | ✅ |

---

# Future Improvements

- Distance-weighted voting
- Manhattan distance
- Minkowski distance
- Cosine distance
- KD-Tree optimization
- Ball Tree optimization
- Radius Neighbors Classifier