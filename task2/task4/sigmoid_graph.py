#4
from typing import Tuple, Union, Optional

import numpy as np
from matplotlib import pyplot as plt
from numpy import ndarray

from task2.task4.sigmoid_gradient import sigmoid_gradient
from sigmoid import sigmoid

def sigmoid_graph():
    tangent_points = [-1, -0.5, 0, 0.5, 1]
    tangent_slopes = sigmoid_gradient(tangent_points)

    x = np.linspace(-2, 2, 100)

    plt.figure()
    plt.plot(x, sigmoid(x), color='blue', label='Сигмоид')
    plt.scatter(tangent_points, sigmoid(tangent_points), color='red')
    for point, slope in zip(tangent_points, tangent_slopes):
        tangent_x: Union[ndarray, Tuple[ndarray, Optional[float]]] = np.linspace(point - 1, point + 1,
                                                                                 10)  # Изменение диапазона x для коротких линий касательных
        tangent_y = slope * (tangent_x - point) + sigmoid(point)
        plt.plot(tangent_x, tangent_y, '--', color='green', linewidth=2)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('График сигмоиды и касательные')
    plt.legend()
    plt.xlim(-2, 2)
    plt.ylim(0, 1)
    plt.axis('equal')
    plt.show()
