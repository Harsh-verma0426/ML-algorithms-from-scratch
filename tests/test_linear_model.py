from mini_sklearn.linear_model import LinearRegression
import numpy as np

def test_fit_runs():
    X = np.array([[1], [2], [3], [4]])
    y = np.array([2, 4, 6, 8])

    model = LinearRegression()

    model.fit(X, y)
    y_pred = model.predict(X)
    assert np.allclose(y, y_pred), "Predictions do not match expected results"
