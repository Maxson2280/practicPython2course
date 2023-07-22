import numpy as np

def add_zero_feature(X):
    return np.insert(X, 0, 1, axis=1)