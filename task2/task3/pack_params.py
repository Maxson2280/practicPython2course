import numpy as np

#3
def pack_params(Theta1, Theta2):
    return np.concatenate((Theta1.ravel(), Theta2.ravel()))
