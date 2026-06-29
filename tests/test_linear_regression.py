import numpy as np

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

def test_intercept_type():

    model = LinearRegression()
    model.fit(X, y)

    assert isinstance(model.intercept_, float)
