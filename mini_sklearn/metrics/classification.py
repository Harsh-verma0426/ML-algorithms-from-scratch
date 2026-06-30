import numpy as np

def accuracy_score(y_test, y_pred):
    return np.mean(y_test==y_pred)