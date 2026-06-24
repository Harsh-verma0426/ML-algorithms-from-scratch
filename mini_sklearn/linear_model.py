'''
Linear Regression And Logistic Regression
----------------------------------------
'''

import numpy as np

class LinearRegression:
    def __init__(self, method='gd', iter=100, learning_rate=0.01):
        # Initialization of Parameters
        self.coef_ = None
        self.intercept_ = 0
        self.method = method
        self.iter = iter
        self.learning_rate = learning_rate

        if self.iter<=0 or self.learning_rate<=0:
            raise ValueError("Iter or learning rate can't be zero or less than zero")
    
    def _ols(self, X, y):
        # Normal equation parts
        XTX = X.T @ X
        XTy = X.T @ y
        # Where W = (X^T * X)^-1 * X^T * y
        self.coef_ = np.linalg.pinv(XTX) @ XTy
        self.intercept_ = np.mean(y - X @ self.coef_)
    
        return self

    def _gd(self, X, y):
        n, d = X.shape
        self.coef_ = np.zeros(d)
        for _ in range(iter):
            y_pred = (X @ self.coef_) + self.intercept_

            dw = (-2/n)* (X.T @ (y-y_pred))
            db = (-2/n)* np.sum(y-y_pred)
            self.coef_ -= dw*self.learning_rate
            self.intercept_ -= db*self.learning_rate

        return self
    
    def fit(self, X, y):
        if self.method=='gd':
            return self._gd(X, y)
        if self.method=='ols':
            return self._ols(X, y)
    
    # Prediction
    def predict(self, X):
        return X @ self.coef_ + self.intercept_




class LogisticRegression:
    def __init__(self, iter=100, learning_rate=0.01):
        self.coef_ = np.array([])
        self.intercept_ = 0
        self.iter = iter
        self.learning_rate = learning_rate

        if self.iter<=0 or self.learning_rate<=0:
            raise ValueError("Iter or learning rate can't be zero or less than zero")

    def sigmoid(z):
        return 1/(1+np.exp(-z))
    
    def fit(self, X, y):
        n = len(X)
        shape = X.shape[1]
        self.coef_ = np.zeros(shape)
        for _ in range(iter):
            z = (X @ self.coef_) + self.intercept_
            y_pred = self.sigmoid(z)
            
            dw = (1/n)* (X.T @ (y_pred-y))
            db = (1/n)* np.sum(y_pred-y)
            self.coef_ -= dw*self.learning_rate
            self.intercept_ -= db*self.learning_rate

        return self

    def predict(self, X, threshold=0.5):
        pred = (X @ self.coef_) + self.intercept_
        return (self.sigmoid(pred)>=threshold).astype(int)