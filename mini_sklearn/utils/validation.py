import numpy as np

def check_numpy_array(array):

    if not isinstance(array, np.ndarray):
        raise TypeError("Expected Array")
    
def check_2d_array(array):

    if array.ndim != 2:
        raise ValueError(f"Expected 2D array, got {array.ndim}D array instead")
    
def check_1d_array(array):

    if array.ndim != 1:
        raise ValueError(f"Expected 1D array, got {array.ndim}D array instead")

def check_same_number_of_samples(X, y):

    if X.shape[0] != y.shape[0]:
        raise ValueError(f"Found input variables with inconsistent numbers of samples: {[X.shape[0], y.shape[0]]}")
    
def check_not_empty(array):

    if array.size == 0:
        raise ValueError("Array is empty")
    
def check_X_y(X, y):

    check_numpy_array(X)
    check_numpy_array(y)
    check_2d_array(X)
    check_1d_array(y)
    check_not_empty(X)
    check_not_empty(y)
    check_same_number_of_samples(X, y)

def check_X(X):

    check_numpy_array(X)
    check_2d_array(X)
    check_not_empty(X)
    
def check_is_fitted(is_fitted):

    if not is_fitted:
        raise ValueError("This estimator is not fitted yet. Call 'fit' before using this estimator.")
    
def check_feature_count(features, X):

    if features != X.shape[1]:
        raise ValueError(
            f"X has {X.shape[1]} features, but the model was fitted with {features} features."
        )