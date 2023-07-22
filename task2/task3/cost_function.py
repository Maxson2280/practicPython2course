import numpy as np

from sigmoid import sigmoid


def cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, Y, lambda_coef):
    # Разделение весов на Theta1 и Theta2
    Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                        (hidden_layer_size, input_layer_size + 1))
    Theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):],
                        (num_labels, hidden_layer_size + 1))

    # Вычисление активаций скрытого слоя
    z2 = np.dot(X, Theta1.T)
    a2 = sigmoid(z2)

    # Добавление столбца смещения к активациям скрытого слоя
    a2 = np.hstack((np.ones((len(X), 1)), a2))

    # Вычисление активаций выходного слоя
    z3 = np.dot(a2, Theta2.T)
    a3 = sigmoid(z3)

    # Вычисление ошибки для каждого примера и класса
    errors = -Y * np.log(a3) - (1 - Y) * np.log(1 - a3)

    # Усреднение ошибки по всем примерам
    J = np.sum(errors) / len(X)

    # Вычисление регуляризации
    reg = (np.sum(Theta1[:, 1:] ** 2) + np.sum(Theta2[:, 1:] ** 2)) * (lambda_coef / (2 * len(X)))

    # Добавление регуляризации к ошибке
    J += reg

    return J
