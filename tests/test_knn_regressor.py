import numpy as np

from mini_sklearn.neighbors import KNeighborsClassifier

X = np.array([[1], [2], [3], [4]])
y = np.array([2, 4, 6, 8])

def test_fit_runs():

    model = KNeighborsClassifier(n_neighbors=2)
    model.fit(X, y)

def test_prediction_shape():

    model = KNeighborsClassifier(n_neighbors=2)
    model.fit(X, y)

    pred = model.predict(X)

    assert pred.shape == (4,)

def test_predict_matches_training_data():

    model = KNeighborsClassifier(n_neighbors=2)

    model.fit(X, y)
    y_pred = model.predict(X)
    assert np.allclose(y, y_pred), "Predictions do not match expected results"