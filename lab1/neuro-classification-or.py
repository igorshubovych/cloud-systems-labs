import numpy as np
import matplotlib.pyplot as plt

k = 50
a = 0
b = 1
step = (b-a)/k

def line(arg):
    return -arg + 0.5

t = a
T = [a]
for i in range(k):
  if t <= b:
    t = t + step
    T.append(t)
print(T)

#ЗАПИСУЄМО ТІЛЬКИ ЗНАЧЕННЯ ФУНКЦІЇ ЯКІ БІЛЬШІ НУЛЯ (МАСИВ y) І ВІДПОВІДНІ ЇМ ЗНАЧЕННЯ АРГУМЕНТУ (МАСИВ Tx)
ch = 0
y = []
Tx = []
for j in range(k):
    f = line(T[j])
    if f >= 0:
        y.append(f)
        ch = j
        Tx.append(T[ch])
print(y)
print(Tx)
plt.plot(Tx, y)
plt.show()

#СТВОРЕННЯ ДВОВИМІРНОГО МАСИВУ ТЕСТОВИХ ДАНИХ (ПАР КООРДИНАТНИХ ТОЧОК)
TEST_DATA = []
POINTS_COUNT = 500
AW = 1
for i in range(POINTS_COUNT):
  x = 0
  test_data = []
  while x <= AW:
    g = np.random.random()*0.8
    test_data.append(g)
    x += 1
  TEST_DATA.append(test_data)
#print(TEST_DATA)

#КЛАСИФІКАЦІЯ
XA = []
YA = []
XB = []
YB = []
def Klass(s1, s2):
    if s1 + s2 -0.5 >0:
        XA.append(s1)
        YA.append(s2)
        return print("klass A")
    else:
        XB.append(s1)
        YB.append(s2)
        return print("klass B")

for i in range(len(TEST_DATA)):
    m = TEST_DATA[i]
    for j in range(len(m)):
        if j + 1 <= len(m) - 1:
            Klass(m[j], m[j + 1])

plt.scatter(XA, YA,  c = 'red')# нанесення контрольних точок на графік
plt.scatter(XB, YB,  c = 'black')# нанесення контрольних точок на графік
plt.show()
