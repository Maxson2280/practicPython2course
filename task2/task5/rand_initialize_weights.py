import numpy as np

def rand_initialize_weights(L_in, L_out):
    # Инициализация случайных весов для матрицы Theta
    epsilon_init = 0.12
    W = np.random.uniform(-epsilon_init, epsilon_init, (L_out, L_in + 1))
    return W
