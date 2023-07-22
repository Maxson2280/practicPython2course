import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat

# from functions import add_zero_feature, sigmoid, decode_y, rand_initialize_weights, pack_params, unpack_params

from sigmoid import sigmoid
from task2.task3.add_zero_feature import add_zero_feature
from task2.task3.decode_y import decode_y
from task2.task3.pack_params import pack_params
from task2.task3.cost_function import cost_function

from task2.task4.sigmoid_gradient import sigmoid_gradient
from task2.task4.sigmoid_graph import sigmoid_graph

from task2.task5.gradient_function import gradient_function
from task2.task5.rand_initialize_weights import rand_initialize_weights
from task2.task5.unpack_params import unpack_params

if __name__ == '__main__':
    # 1
    test_set = loadmat('training_set.mat')
    X = test_set['X']
    y = test_set['y']

    weights = loadmat('weights.mat')
    Theta1 = weights['Theta1']
    Theta2 = weights['Theta2']
    Theta2 = weights['Theta2']

    # 2
    input_layer_size = X.shape[1]  # количество входов сети (20*20=400)
    hidden_layer_size = Theta1.shape[0]  # нейронов в скрытом слое(25)
    num_labels = len(np.unique(y))  # число распознаваемых классов(10)
    m = len(X)  # количество примеров(5000)

    # 3

    # добавление единичного столбца - нейрон смещения
    X = add_zero_feature(X)

    # декодирование вектора Y
    Y = decode_y(y)

    # объединение матриц Theta в один большой массив
    nn_params = pack_params(Theta1, Theta2)

    # проверка функции стоимости для разных lambda
    lambda_coef = 0
    print('Функция стоимости для lambda {} = {}'.
          format(lambda_coef, cost_function(
        nn_params, input_layer_size, hidden_layer_size,
        num_labels, X, Y, lambda_coef)))

    lambda_coef = 1
    print('Функция стоимости для lambda {} = {}'.
          format(lambda_coef, cost_function(
        nn_params, input_layer_size, hidden_layer_size,
        num_labels, X, Y, lambda_coef)))

    # 4
    # проверка производной sigmoid
    gradient = sigmoid_gradient(np.array([-1, -0.5, 0, 0.5, 1]))
    print('Производная функции sigmoid в точках -1, -0.5, 0, 0.5, 1:')
    print(gradient)

    sigmoid_graph()

    # 5
    # случайная инициализация параметров
    initial_Theta1 = rand_initialize_weights(input_layer_size, hidden_layer_size)
    initial_Theta2 = rand_initialize_weights(hidden_layer_size, num_labels)
    initial_nn_params = pack_params(initial_Theta1, initial_Theta2)

    # обучение нейронной сети
    res = minimize(cost_function, initial_nn_params, method='L-BFGS-B',
                   jac=gradient_function, options={'maxiter': 100},
                   args=(input_layer_size, hidden_layer_size, num_labels, X, Y, lambda_coef)).x

    # разбор вычисленных параметров на матрицы Theta1 и Theta2
    Theta1, Theta2 = unpack_params(
        res, input_layer_size, hidden_layer_size, num_labels)

    # выичисление отклика сети на примеры из обучающей выборки
    h1 = sigmoid(np.dot(X, Theta1.T))

    h2 = sigmoid(np.dot(add_zero_feature(h1), Theta2.T))

    y_pred = np.argmax(h2, axis=1) + 1

print('Точность нейронной сети на обучающей выборке: {}'.format(np.mean(y_pred == y.ravel()) * 100))
