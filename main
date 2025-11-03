import numpy as np
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt

h = 1
r = h ** 2 * 0.95  # r < h^2
N = 20  # количества шагов по осям
J = 25  # по разным осям беру разные значения, чтобы убедиться, что ни где в коде их не перепутаю
A = 2

# Начальные вектора
J0x = np.zeros(J)  # Исходные значения при N = 0
Jx0 = np.zeros(N)  # Исходные значения при J = 0


def F(P):
    return np.cos(P) * (1 + A * np.sin(P) ** 2)


'''
    Equation - номер уравнения (начиная с 0)
    N   - количество шагов по оси N
    J   - количество шагов по оси J
    J0x - вектор начальных значений Ф(n = 0)
    Jx0 - вектор начальных значений Ф(j = 0) или j = J-1 при Equation = 1
    Jx02- вектор начальных значений Ф(j = J-1) используется лишь при Equation = 2. Если не указан, используем J0x
'''


def Calc(r, h, N, J, J0x, Jx0, Equation, Jx02=None):
    global fig

    # создаем матрицу для результата
    # При этом учитываем, что если в формуле используетя J+1, нам надо рассчитывать на 1 элемент более того, что будет возвращать
    Z = np.zeros((N, J), dtype=np.float32)

    # устанавливаем начальные значения
    # Для разных уравнений разные начальные настройки
    if Equation == 0:
        Z[0, :] = J0x
        Z[:, 0] = Jx0

    elif Equation == 1:
        Z[0, :] = J0x
        Z[:, -1] = Jx0

    else:
        Z[0, :] = J0x
        Z[:, 0] = Jx0
        Z[:, -1] = Jx02 if Jx02 is not None else Jx0

    for n in range(N - 1):  # в явном виде цикла по J мы не делаем. Вместо этого осуществляем векторные вычисления -
        # всю строку за один раз. Так эффективнее
        Jn = Z[n]

        # Проводим пакетные вычисления сразу для всей строки
        # Для разный уравнений - разные алгоритмы
        if Equation == 0:
            Jn0 = Jn[1:]
            Jn1 = Jn[:-1]
            dJn = (Jn0 - Jn1) / h  # находим  (Ф(j,n) - Ф(j-1,n))/h
        elif Equation == 1:
            Jn0 = Jn[2:]
            Jn1 = Jn[1:-1]
            dJn = (Jn0 - Jn1) / h  # находим  (Ф(j,n) - Ф(j-1,n))/h
        elif Equation == 2:
            Jn0 = Jn[2:]
            Jn1 = Jn[:-2]
            dJn = (Jn0 - Jn1) / (2 * h)  # находим  (Ф(j,n) - Ф(j-1,n))/h

        SQRT = np.sqrt(dJn ** 2 + 1)  # находим то, что под корнем
        ArcTn = np.arctan(dJn)  # находим аргтангенс
        f = F(ArcTn)  # Находим F(Ф)
        Res = -r * f * SQRT + Jn0  # Заполняем всю строку n+1, кроме первого значения, которое относится к Ф0 и не вычисляется

        if Equation == 0:
            Z[n + 1, 1:] = Res
        elif Equation == 1:
            Z[n + 1, 1:-1] = Res  # находим  (Ф(j,n) - Ф(j-1,n))/h
        elif Equation == 2:
            Z[n + 1, 1:-1] = Res

    n_points = np.arange(0, r * N, r)
    j_points = np.arange(0, h * J, h)

    X, Y = np.meshgrid(j_points, n_points)

    fg = plt.figure()

    ax = fg.add_subplot(111, projection='3d')

    ax.plot_surface(X, Y, Z, cmap='jet')
    ax.set_title(f'Ф(n,j) уравнение {Equation}')
    ax.set_xlabel('n')
    ax.set_ylabel('h')
    ax.set_zlabel('Ф')



Calc(r, h, N, J, J0x, Jx0, 0)
Calc(r, h, N, J, J0x, Jx0, 1)
Calc(r, h, N, J, J0x, Jx0, 2)

plt.show()

