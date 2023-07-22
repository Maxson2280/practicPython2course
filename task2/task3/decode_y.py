import numpy as np

def decode_y(y):
    num_labels = len(np.unique(y))
    Y = np.zeros((len(y), num_labels))
    Y[np.arange(len(y)), y.ravel() - 1] = 1
    return Y
