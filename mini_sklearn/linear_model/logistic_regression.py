import numpy as np
from ..utils import (
    check_X_y,
    check_X,
    check_is_fitted,
    check_feature_count
)

class LogisticRegression:
    def __init__(self, maxiter=1000, learning_rate=0.01):
        self.coef_ = np.array([])
        self.intercept_ = 0
        self.maxiter = maxiter
        self.learning_rate = learning_rate
        self.n_features_in_ = None
        self.is_fitted_ = False

        if self.maxiter<=0 or self.learning_rate<=0:
            raise ValueError("Iter or learning rate can't be zero or less than zero")

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))
    
    def fit(self, X, y):

        check_X_y(X, y)

        n = len(X)
        shape = X.shape[1]
        self.coef_ = np.zeros(shape)
        for _ in range(self.maxiter):
            z = (X @ self.coef_) + self.intercept_
            y_pred = self.sigmoid(z)
            
            dw = (1/n)* (X.T @ (y_pred-y))
            db = (1/n)* np.sum(y_pred-y)
            self.coef_ -= dw*self.learning_rate
            self.intercept_ -= db*self.learning_rate

        self.n_features_in_ = X.shape[1]
        self.is_fitted_ = True

        return self

    def predict(self, X, threshold=0.5):

        check_X(X)
        check_feature_count(self.n_features_in_, X)
        check_is_fitted(self.is_fitted_)

        pred = (X @ self.coef_) + self.intercept_
        return (self.sigmoid(pred)>=threshold).astype(int)