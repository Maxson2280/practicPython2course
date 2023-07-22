import numpy as np

def unpack_params(params, input_layer_size, hidden_layer_size, num_labels):
    # Разбиение параметров на матрицы Theta1 и Theta2
    num_elems_theta1 = hidden_layer_size * (input_layer_size + 1)
    Theta1 = np.reshape(params[:num_elems_theta1], (hidden_layer_size, input_layer_size + 1))
    Theta2 = np.reshape(params[num_elems_theta1:], (num_labels, hidden_layer_size + 1))
    return Theta1, Theta2
