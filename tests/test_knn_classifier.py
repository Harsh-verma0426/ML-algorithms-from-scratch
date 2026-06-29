import numpy as np

from mini_sklearn.neighbors import KNeighborsClassifier

X = np.array([[1], [2], [3], [4], [5], [6]])
y = np.array([0, 0, 0, 1, 1, 1])


def test_fit_runs():
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X, y)

def test_predict_matches_training_data():

    # maxiter is needed here because default maxiter is 1000 which is low for accurate prediction
    # lbfgs which sklearn uses is future score
    model = KNeighborsClassifier(n_neighbors=3) 

    model.fit(X, y)
    y_pred = model.predict(X)
    assert np.allclose(y, y_pred), "Predictions do not match expected results"

def test_prediction_shape():

    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X, y)

    pred_ = model.predict(X)

    assert pred_.shape == (6,)