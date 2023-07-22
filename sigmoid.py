import numpy as np
#4
def sigmoid(z):
    z = np.array(z)
    g = 1 / (1 + np.exp(-z))
    return g
