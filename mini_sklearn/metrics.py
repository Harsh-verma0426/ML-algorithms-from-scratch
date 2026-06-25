import numpy as np

def r2_score(y_test, y_pred):
    ssr = np.sum((y_test-y_pred)**2)
    sst = np.sum((y_test - np.mean(y_test))**2)
    return 1-(ssr/sst)


def accuracy_score(y_test, y_pred):
    return np.mean(y_test==y_pred)