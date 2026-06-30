import numpy as np

def train_test_split(
                X,
                y,
                test_size=0.2,
                train_size=None,
                random_state=None,
                shuffle=True
            ):
    
    X = np.asarray(X)
    y = np.asarray(y)
    n = len(X)
    indices = np.arange(n)
    if random_state is not None:
        np.random.seed(random_state)
    if shuffle:
        np.random.shuffle(indices)

    if train_size is not None:
        test_size = 1-train_size
    split_idx = int(n*(1-test_size))

    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    return X_train, X_test, y_train, y_test