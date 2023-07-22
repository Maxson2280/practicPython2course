import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat
from displayData import displayData
from predict import predict

# Загрузка данных и весов
test_set = loadmat('test_set.mat')
X = test_set['X']
y = test_set['y']

weights = loadmat('weights.mat')
Theta1 = weights['Theta1']
Theta2 = weights['Theta2']

# Выбор случайных примеров для отображения
m = X.shape[0]
indices = np.random.permutation(m)[:100]
selected_examples = X[indices, :]

# Отображение случайных примеров
displayData(selected_examples)

# Предсказание меток классов
pred = predict(Theta1, Theta2, X)
pred = np.argmax(pred, axis=1) + 1

# Расчет точности
accuracy = np.mean(pred == y.ravel()) * 100
print(f"Точность: {accuracy}%")

# Классификация 5 случайных примеров
rp = np.random.permutation(m)
plt.figure()
for i in range(5):
    X2 = X[rp[i], :]
    X2 = np.matrix(X[rp[i]])

    pred = predict(Theta1, Theta2, X2.getA())
    pred = np.argmax(pred, axis=1) + 1
    pred = np.squeeze(pred)

    pred_str = 'Neural Network Prediction: %d (digit %d)' % (pred, y[rp[i]])
    displayData(X2, pred_str)

# Отображение ошибочных примеров

incorrect_indices = np.where(pred != y.ravel())[1]
displayData(X[incorrect_indices[:100]], "Error Neural Network")

