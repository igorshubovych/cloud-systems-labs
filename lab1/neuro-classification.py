import numpy as np
import matplotlib.pyplot as plt
#import scipy
#from numpy import random

k = 50
a = 0
b = 1
step = (b-a)/k


def line(arg):
    return -arg + 1.2

t = a
T = [a]
for i in range(k):
  if t <= b:
    t = t + step
    T.append(t)
print(T)

# ЗАПИСУМО ТІЛЬКИ ЗНАЧЕННЯ ФУНКЦІЇ ЯКІ БІЛЬШІ НУЛЯ (МАСИВ y)
# І ВІДПОВІДНІ ЇМ ЗНАЧЕННЯ АРГУМЕНТУ (МАСИВ Tx)
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

# СТВОРЕННЯ ДВОВИМІРНОГО МАСИВУ FUNC ПАР КООРДИНАТНИХ ТОЧОК
FUNC = []
CR = 500
AW = 1
for i in range(CR):
  x = 0
  func = []
  while x <= AW:
    g = np.random.random()*0.8
    func.append(g)
    x += 1
  FUNC.append(func)
#print(FUNC)

# КЛАСИФІКАЦІЯ
X1 = []
Y1 = []
X11 = []
Y11 = []
def Klass(s1, s2):
    if s1 + s2 -0.5 >0:
        X1.append(s1)
        Y1.append(s2)
        return print("klass A")
    else:
        X11.append(s1)
        Y11.append(s2)
        return print("klass B")

for i in range(len(FUNC)):
    m = FUNC[i]
    for j in range(len(m)):
        if j + 1 <= len(m) - 1:
            Klass(m[j], m[j + 1])

plt.scatter(X1, Y1,  c = 'red')# нанесення контрольних точок на графік
plt.scatter(X11, Y11,  c = 'black')# нанесення контрольних точок на графік
plt.show()

