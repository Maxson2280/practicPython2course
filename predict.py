import numpy as np
from sigmoid import sigmoid
#5
def predict(Theta1, Theta2,X):

    # Добавление единичного столбца к матрице X
    m = X.shape[0]
    ones = np.ones((m, 1))
    a1 = np.c_[ones, X]

    # Вычисление значений на входах скрытого слоя (a2)
    g1 = np.dot(a1, Theta1.T)
    a2 = sigmoid(g1)

    # Добавление единичного столбца к матрице a2
    ones = np.ones((a2.shape[0], 1))
    a2 = np.c_[ones, a2]

    # Вычисление значений на выходе скрытого слоя (a3 или a2 пропущенное через sigmoid)
    g3 = np.dot(a2, Theta2.T)
    a3 = sigmoid(g3)

    return a3
