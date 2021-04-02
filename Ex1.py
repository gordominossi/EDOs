import numpy as np
from common import *
import math

x0 = np.array([1, 1, 1, -1])

A = np.matrix([[-2, -1, -1, -2],
               [1, -2, 2, -1],
               [-1, -2, -2, -1],
               [2, -1, 1, -2]])

F = []
for i in range(len(A)):
    def f(time: float, x: np.ndarray) -> np.ndarray:
        sum = 0
        for j in range(len(A[i])):
            sum += A[i, j] * x[i]
        return sum
    F.append(f)

phiRK44 = generateRK44Phi(Function(F))

print(solveEDO(x0, phiRK44, [0, 2], 20))
exactF = [
    lambda time: math.exp(-time) * math.sin(time) + math.exp(-3 * time) * math.cos(3 * time),
    lambda time: math.exp(-time) * math.cos(time) + math.exp(-3 * time) * math.sin(3 * time),
    lambda time: -math.exp(-time) * math.sin(time) + math.exp(-3 * time) * math.cos(3 * time),
    lambda time: -math.exp(-time) * math.cos(time) + math.exp(-3 * time) * math.sin(3 * time)
]

print(getSolution(exactF, [0, 2], 20))
