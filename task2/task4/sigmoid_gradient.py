#4
from sigmoid import sigmoid
#4
def sigmoid_gradient(z):
    g = sigmoid(z)*(1 - sigmoid(z))
    return g