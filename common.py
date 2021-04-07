from typing import Callable, List
import numpy as np
from decimal import Decimal

MAX_NEWTON_ITERATIONS = 7
phiType = Callable[[float, np.ndarray, float], np.ndarray]
fType = Callable[[float, np.ndarray], np.ndarray]
jacobianType = List[List[Callable[[float, float, np.ndarray], float]]]
matrixType = np.ndarray


class Function:
    def __init__(self, fArray: List[fType]):
        self.fArray = fArray
        for i in range(len(fArray)):
            print(fArray[i](0, np.array([1, 1, 1, -1])))

    def __call__(self, time: float, U: np.ndarray) -> np.ndarray:
        result = np.empty(len(self.fArray))
        for i in range(len(self.fArray)):
            result[i] = self.fArray[i](time, U)
        return result


def solveEDO(u0: np.ndarray, phi: phiType, interval: np.ndarray, discretization: int):
    U = np.empty((discretization + 1, len(u0)))
    U[0] = u0
    step = (interval[1] - interval[0]) / discretization
    for iterations in range(discretization):
        U[iterations + 1] = U[iterations] + \
            step * phi(interval[0] + step * iterations, U[iterations], step)
    return U


def getSolution(exactF: fType, interval: np.ndarray, discretization: int):
    X = np.empty((discretization + 1, len(exactF)))
    step = (interval[1] - interval[0]) / discretization
    for iterations in range(discretization + 1):
        for i in range(len(exactF)):
            X[iterations, i] = exactF[i](interval[0] + iterations * step)
    return X


def generateRK44Phi(f: Function) -> phiType:
    def phi(time: float, currentU: np.ndarray, step: float) -> np.ndarray:
        kappa1 = f(time, currentU)
        kappa2 = f(time + step / 2, currentU + step * kappa1 / 2)
        kappa3 = f(time + step / 2, currentU + step * kappa2 / 2)
        kappa4 = f(time + step, currentU + step * kappa3)
        return (kappa1 + 2 * kappa2 + 2 * kappa3 + kappa4) / 6
    return phi


def generateImplicitEulerPhi(f: Function, J: jacobianType) -> phiType:
    def phi(time: float, currentU: np.ndarray, step: float) -> np.ndarray:
        def generateG(currentU: np.ndarray) -> Function:
            def g(time: float, nextU: np.ndarray):
                return nextU - step * f(time, nextU) - currentU
            return g

        def generateJacobian(time: float, nextU: np.ndarray, currentU: np.ndarray) -> matrixType:
            jacobian = np.empty((len(J), len(J)))
            for i in range(len(J)):
                line = np.empty(len(J[i]))
                for j in range(len(J[i])):
                    line[j] = J[i][j](time, step, *nextU)
                jacobian[i] = line
            return jacobian

        def inverseJacobian(time: float, nextU: np.ndarray, currentU: np.ndarray) -> matrixType:
            return np.linalg.inv(generateJacobian(time, nextU, currentU))

        def newtonIteration(currentNextUAproximation: np.ndarray, previousNextUAproximation: np.ndarray):
            g = generateG(previousNextUAproximation)
            return currentNextUAproximation - inverseJacobian(time, currentNextUAproximation, previousNextUAproximation) * g(time + step, currentNextUAproximation)

        nextU = previousNextUAproximation = currentNextUAproximation = currentU
        for dummyIterationCounter in range(MAX_NEWTON_ITERATIONS):
            previousNextUAproximation = currentNextUAproximation
            currentNextUAproximation = nextU
            nextU = newtonIteration(currentNextUAproximation,
                                    previousNextUAproximation)
        return nextU
    return phi
