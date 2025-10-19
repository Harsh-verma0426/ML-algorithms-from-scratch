'''
Linear Regression using Gradient Descent
----------------------------------------
This script implements simple linear regression from scratch
using gradient descent, without sklearn.

'''

import numpy as np

class LinearRegressionGD:
    def __init__(self):
        # Initialize slope and intercept.
        self.m = 0
        self.c = 0

    def iteration(self,epoches):
        if epoches < 0:
            raise ValueError("Iteration must be positive")
        self.epoches = epoches
        return self  # Required for chaining

    def learning_rate(self,alpha):
        if alpha <= 0:
            raise ValueError("Learning rate must be higher than 0")
        self.alpha = alpha
        return self  # Required for chaining
    
    def fit(self,x,y):
        x = np.array(x)
        y = np.array(y)

        n = len(x)

        for i in range(self.epoches):
            y_pred = self.m*x + self.c

            dm = (-2/n) * sum(x*(y-y_pred))
            dc = (-2/n) * sum(y-y_pred)

            self.m -= self.alpha*dm
            self.c -= self.alpha*dc    

        return self  # Required for chaining
    
    # Prediction
    def predict(self, x):
        import numpy as np
        x = np.array(x)
        return self.m * x + self.c
    
    def parameters(self):
        return {"slope": self.m, "intercept": self.c}


if __name__  == "__main__":
    model = LinearRegressionGD()
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 3, 5, 7, 11])
    model.learning_rate(0.001).iteration(100000).fit(x,y)
    print(model.parameters())
    print(model.predict(6))
