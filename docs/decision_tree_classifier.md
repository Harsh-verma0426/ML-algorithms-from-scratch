# Decision Tree Classifier

## Overview

Decision Tree is a supervised machine learning algorithm used for classification and regression tasks. It learns a hierarchy of decision rules by recursively splitting the training data based on feature values.

This implementation builds a binary classification tree using **Information Gain** as the splitting criterion and **Entropy** as the impurity measure.

---

# How It Works

The algorithm follows these steps:

1. Compute the entropy of the current node.
2. Evaluate every possible split across all features.
3. Calculate the Information Gain for each split.
4. Select the split with the highest Information Gain.
5. Divide the dataset into two subsets.
6. Repeat the process recursively for each child node.
7. Stop when a leaf node is reached.

---

# Entropy

Entropy measures the impurity or randomness of a dataset.

```text
Entropy = -Σ pᵢ log₂(pᵢ)
```

Properties:

- Entropy = 0 → Pure node
- Higher entropy → More mixed classes
- Maximum entropy occurs when classes are equally distributed.

---

# Information Gain

Information Gain measures how much uncertainty is reduced after a split.

```text
Information Gain = Parent Entropy − Weighted Child Entropy
```

A larger Information Gain indicates a better split.

---

# Splitting Criterion

For every feature:

1. Find all unique values.
2. Use each value as a candidate threshold.
3. Split the dataset into:

```text
Left  : feature ≤ threshold
Right : feature > threshold
```

4. Compute the Information Gain.
5. Keep the split with the highest gain.

---

# Stopping Criteria

Tree construction stops when:

- All samples belong to the same class.
- No split provides positive Information Gain.
- A split produces an empty child node.

In these cases, the node becomes a leaf.

---

# Prediction

Prediction starts from the root node.

For each internal node:

```text
feature ≤ threshold ?
```

- Yes → Traverse left child.
- No → Traverse right child.

Repeat until a leaf node is reached.

The class stored in the leaf node is returned as the prediction.

---

# Implementation Details

Current implementation includes:

- Binary classification
- Entropy impurity measure
- Information Gain split criterion
- Recursive tree construction
- Binary tree traversal for prediction
- Automatic leaf node creation
- Majority class prediction when no useful split exists

---

# Computational Complexity

| Operation | Complexity |
|----------|------------|
| Fit | O(n_features × n_samples²) *(naive implementation)* |
| Predict | O(tree depth) per sample |

---

# Example

```python
from mini_sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()

model.fit(X_train, y_train)

predictions = model.predict(X_test)
```

---

# Comparison with scikit-learn

| Feature | This Project | scikit-learn |
|----------|--------------|--------------|
| Classification | ✅ | ✅ |
| Entropy Criterion | ✅ | ✅ |
| Recursive Tree Construction | ✅ | ✅ |
| Binary Prediction | ✅ | ✅ |
| Gini Criterion | ❌ | ✅ |
| Pruning | ❌ | ✅ |
| max_depth | ❌ | ✅ |
| min_samples_split | ❌ | ✅ |
| Feature Importance | ❌ | ✅ |

---

# Future Improvements

- Gini Impurity criterion
- Maximum tree depth
- Minimum samples required for splitting
- Minimum samples per leaf
- Cost-complexity pruning
- Random feature selection
- Feature importance calculation
- Decision tree visualization
- Support for regression trees