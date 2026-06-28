# ML Algorithms

A lightweight Python library implementing classical machine learning algorithms from first principles using **NumPy**. The goal of this project is to understand how machine learning algorithms work internally while maintaining an API inspired by scikit-learn.

> **Note:** This project is intended for learning and educational purposes rather than production use.

---

## Features

### Linear Models

* ✅ Linear Regression

  * Ordinary Least Squares (OLS)
  * Gradient Descent
* ✅ Logistic Regression

### Nearest Neighbors

* ✅ K-Nearest Neighbors Regressor
* ✅ K-Nearest Neighbors Classifier

### Decision Trees

* ✅ Decision Tree Classifier

### Metrics

* ✅ R² Score
* ✅ Accuracy Score

### Model Selection

* ✅ Train-Test Split

---

## Project Structure

```text
mini_sklearn/
│
├── linear_model.py
├── neighbors.py
├── tree.py
├── metrics.py
├── model_selection.py
└── __init__.py

tests/
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Harsh-verma0426/ML-algorithms-from-scratch.git
cd ML-algorithms-from-scratch
```

Create a virtual environment:

```bash
uv venv
```

Install dependencies:

```bash
uv sync
```

---

## Quick Example

```python
from mini_sklearn.linear_model import LinearRegression
from mini_sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = LinearRegression(method="gd")
model.fit(X_train, y_train)

predictions = model.predict(X_test)
score = model.score(y_test, predictions)

print(score)
```

---

## Roadmap

### Core Algorithms

* [x] Linear Regression
* [x] Logistic Regression
* [x] KNN Regressor
* [x] KNN Classifier
* [x] Decision Tree
* [ ] Random Forest
* [ ] Naive Bayes
* [ ] K-Means
* [ ] PCA
* [ ] Ridge Regression
* [ ] Lasso Regression
* [ ] Support Vector Machine (SVM)

### Utilities

* [x] Train-Test Split
* [ ] StandardScaler
* [ ] MinMaxScaler
* [ ] LabelEncoder
* [ ] OneHotEncoder
* [ ] K-Fold Cross Validation

### Project Improvements

* [ ] Comprehensive documentation
* [ ] Unit tests for all algorithms
* [ ] Benchmark against scikit-learn
* [ ] Example notebooks

---

## References

The implementations and concepts in this project are based on the following resources:

* Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow — Aurélien Géron
* NumPy Documentation
* scikit-learn Documentation
