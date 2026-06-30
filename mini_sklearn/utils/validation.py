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

def check_number_of_samples_different(X, y):

    if X.shape[0] != y.shape[0]:
        raise ValueError(f"Found input variables with inconsistent numbers of samples: {[X.shape[0], y.shape[0]]}")
    
def check_array_empty(array):

    if array.size == 0:
        raise ValueError("Array is empty")
    
def check_X_y(X, y):

    check_numpy_array(X)
    check_numpy_array(y)
    check_2d_array(X)
    check_1d_array(y)
    check_array_empty(X)
    check_array_empty(y)
    check_number_of_samples_different(X, y)

def check_X(X):

    check_numpy_array(X)
    check_2d_array(X)
    check_array_empty(X)

def check_maxiter_learning_rate(maxiter, learning_rate):

    if maxiter<=0 or learning_rate<=0:
        raise ValueError("Iter or learning rate can't be zero or less than zero")
    
