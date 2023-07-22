import numpy as np
import matplotlib.pyplot as plt

def generate_pareto(alpha, size):
    return (np.random.uniform(0, 1, size) ** (-1/alpha))

def pareto_probability(x, alpha):
    return alpha / (x ** (alpha + 1))

def pareto_distribution(x, alpha):
    return 1 - (1 / (x ** alpha))

# Генерация псевдослучайных значений
alpha = 2.5
size = 1000
data = generate_pareto(alpha, size)
print(data)
# Оценивание параметра методом максимального правдоподобия
estimated_alpha = size / np.sum(np.log(data))
print("Оценка параметра alpha:", estimated_alpha)

# Построение графика функции вероятности
x = np.linspace(np.min(data), np.max(data), 100)
pdf = pareto_probability(x, alpha)
plt.plot(x, pdf)
plt.title("Функция вероятности распределения Парето")
plt.xlabel("Значение")
plt.ylabel("Вероятность")
plt.show()

# Построение графика функции распределения
cdf = pareto_distribution(x, alpha)
plt.plot(x, cdf)
plt.title("Функция распределения Парето")
plt.xlabel("Значение")
plt.ylabel("Вероятность")
plt.show()

#гистограмм по сгенерированному набору чисел

plt.hist(data, bins=100, density=True)
plt.xlabel('Значение')
plt.ylabel('Частота')
plt.title('Гистограмма сгенерированных чисел')
plt.show()
