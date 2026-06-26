import numpy as np

def train_test_split(X, y, test_size):

    n = len(X)
    indices = np.arange(n)
    np.random.shuffle(indices)

    split_idx = int(n*(1-test_size))

    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    return X_train, X_test, y_train, y_test