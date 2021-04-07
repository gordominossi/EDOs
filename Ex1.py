# from common import Function, generateRK44Phi, generateImplicitEulerPhi, solveEDO, getSolution
# import numpy as np
# import math


def RK44():
    x0 = [+1, +1, +1, -1]
    interval = [0, 2]
    n = 20
    A = [[-2, -1, -1, -2],
         [+1, -2, +2, -1],
         [-1, -2, -2, -1],
         [+2, -1, +1, -2]]

    def dot(A, x):
        s = 0
        for i in range(len(A)):
            s += A[i] * x[i]
        return s

    F = []
    for i in range(len(A)):
        def f(t, x):
            return dot(A[i], x)
        F.append(f)

    for i in range(len(F)):
        print(F[i](0, x0))
    print()

    for f in F:
        print(f(0, x0))
    print()

    # Function(F)(0, x0)
    # phiRK44 = generateRK44Phi(Function(F))
    # print(solveEDO(x0, phiRK44, interval, n), "\n")

    # exactF = [
    #     lambda time: math.exp(-time) * math.sin(time) +
    #     math.exp((-3) * time) * math.cos(3 * time),
    #     lambda time: math.exp(-time) * math.cos(time) +
    #     math.exp((-3) * time) * math.sin(3 * time),
    #     lambda time: -math.exp(-time) * math.sin(time) +
    #     math.exp((-3) * time) * math.cos(3 * time),
    #     lambda time: -math.exp(-time) * math.cos(time) +
    #     math.exp((-3) * time) * math.sin(3 * time)
    # ]
    # print(getSolution(exactF, interval, n), "\n")


RK44()


# def implicitEuler():
#     x0 = np.array([-8.79])
#     interval = [1.1, 3.0]
#     n = 5000

#     F = [lambda t, u: 2 * t + (u - t ** 2) ** 2]

#     J = [[lambda t, h, u: 1 - h * F[0](t, u)]]

#     phiImplicitEuler = generateImplicitEulerPhi(Function(F), J)
#     print(solveEDO(x0, phiImplicitEuler, interval, n)[n // 2])

#     exactF = [lambda t: t ** 2 + 1 / (1 - t)]
#     print(getSolution(exactF, interval, n)[n // 2])


# implicitEuler()
