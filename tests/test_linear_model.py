import numpy as np

#----------------------Linear Regression-------------------------------

from mini_sklearn.linear_model import LinearRegression

X = np.array([[1], [2], [3], [4]])
y = np.array([2, 4, 6, 8])

def test_fit_runs():

    model = LinearRegression()
    model.fit(X, y)

def test_predict_matches_training_data():

    model = LinearRegression()

    model.fit(X, y)
    y_pred = model.predict(X)
    assert np.allclose(y, y_pred), "Predictions do not match expected results"

def test_coef_shape():

    model = LinearRegression()
    model.fit(X, y)

    assert model.coef_.shape == (1,)

def test_prediction_shape():

    model = LinearRegression()
    model.fit(X, y)

    pred = model.predict(X)

    assert pred.shape == (4,)

#----------------------Logistic Regression-------------------------------

from mini_sklearn.linear_model import LogisticRegression

X = np.array([[1], [2], [3], [4], [5], [6]])
y = np.array([0, 0, 0, 1, 1, 1])


def test_fit_runs():
    model = LogisticRegression()
    model.fit(X, y)

def test_predict_matches_training_data():

    # maxiter is needed here because default maxiter is 1000 which is low for accurate prediction
    # lbfgs which sklearn uses is future score
    model = LogisticRegression(maxiter=10000) 

    model.fit(X, y)
    y_pred = model.predict(X)
    assert np.allclose(y, y_pred), "Predictions do not match expected results"

def test_prediction_shape():

    model = LogisticRegression()
    model.fit(X, y)

    pred = model.predict(X)

    assert pred.shape == (6,)

def test_coef_shape():

    model = LogisticRegression()
    model.fit(X, y)

    assert model.coef_.shape == (1,)
