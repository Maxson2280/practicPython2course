import numpy as np
from functions import add_zero_feature, sigmoid, decode_y, unpack_params, pack_params
from task2.task4.sigmoid_gradient import sigmoid_gradient


def gradient_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, Y, lambda_coef):
    m = X.shape[0]  # Количество примеров в выборке

    # Разбор параметров на матрицы Theta1 и Theta2
    Theta1, Theta2 = unpack_params(nn_params, input_layer_size, hidden_layer_size, num_labels)

    # Вычисление отклика нейронной сети для всех примеров из выборки
    a1 = X
    z2 = np.dot(a1, Theta1.T)
    a2 = sigmoid(z2)
    a2 = add_zero_feature(a2)
    z3 = np.dot(a2, Theta2.T)
    a3 = sigmoid(z3)

    # Вычисление ошибок в третьем слое (выходном)
    d3 = a3 - Y

    # Вычисление ошибок во втором слое (скрытом)
    d2 = np.dot(d3, Theta2) * sigmoid_gradient(add_zero_feature(z2))
    d2 = d2[:, 1:]

    # Вычисление частных производных для градиентного спуска
    Delta1 = np.dot(d2.T, a1)
    Delta2 = np.dot(d3.T, a2)

    # Учет регуляризации
    Theta1_reg = np.copy(Theta1)
    Theta1_reg[:, 0] = 0
    Theta2_reg = np.copy(Theta2)
    Theta2_reg[:, 0] = 0

    Theta1_grad = (Delta1 + lambda_coef * Theta1_reg) / m
    Theta2_grad = (Delta2 + lambda_coef * Theta2_reg) / m

    # Объединение градиентов в один вектор
    grad = pack_params(Theta1_grad, Theta2_grad)

    return grad
