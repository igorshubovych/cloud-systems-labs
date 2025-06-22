import numpy as np
import matplotlib.pyplot as plt

# Параметри
POINTS_COUNT = 500  # Кількість точок
THRESHOLD = 0.5  # Поріг для активаційної функції

# Функція OR
def perceptron_or(x1, x2):
    return x1 + x2 > THRESHOLD

# Функція AND
def perceptron_and(x1, x2):
    return x1 > THRESHOLD and x2 > THRESHOLD

# Функція NOT
def perceptron_not(x):
    return not x

# Функція XOR
def perceptron_xor(x1, x2):
    y1 = perceptron_or(x1, x2)
    y2 = perceptron_and(x1, x2)
    return perceptron_and(y1, perceptron_not(y2))

# Генерація випадкового набору даних
TEST_DATA = np.random.random((POINTS_COUNT, 2))

# Результати класифікації
XA = []
YA = []
XB = []
YB = []

for x1, x2 in TEST_DATA:
    if perceptron_xor(x1, x2):
        XA.append(x1)
        YA.append(x2)
    else:
        XB.append(x1)
        YB.append(x2)

# Візуалізація результатів
plt.scatter(XA, YA, c='red', label='Class A (XOR=1)')
plt.scatter(XB, YB, c='black', label='Class B (XOR=0)')
plt.legend()
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Класифікація XOR за допомогою нейронної мережі')
plt.show()