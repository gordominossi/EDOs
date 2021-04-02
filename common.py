from typing import Callable, List
import numpy as np
from decimal import Decimal

MAX_NEWTON_ITERATIONS = 10
phiType = Callable[[float, np.ndarray, float], np.ndarray]
fType = Callable[[float, np.ndarray], np.ndarray]
matrixType = np.ndarray


class Function:
    def __init__(self, fArray: List[fType]):
        self.fArray = fArray

    def __call__(self, time: float, u: np.ndarray) -> np.ndarray:
        result = np.empty(len(u))
        for i in range(len(u)):
            result[i] = self.fArray[i](time, u)
        return result

    def getFs(self) -> np.ndarray:
        return self.fArray


def solveEDO(u0: np.ndarray, phi: phiType, interval: np.ndarray, discretization: int):
    U = np.empty((discretization + 1, len(u0)))
    U[0] = u0
    step = (interval[1] - interval[0]) / discretization
    for iterations in range(discretization):
        U[iterations + 1] = iteration(interval[0] + step * iterations,
                                      U[iterations], step, phi)
    return U


def getSolution(exactF: fType, interval: np.ndarray, discretization: int):
    X = np.empty((discretization, len(exactF) + 1))
    step = (interval[1] - interval[0]) / discretization
    for iterations in range(discretization):
        for i in range(len(exactF)):
            X[iterations, i] = exactF[i](interval[0] + iterations * step)
    return X


def iteration(time: float, currentU: np.ndarray, step: float, phi: phiType) -> np.ndarray:
    nextU = currentU + step * phi(time, currentU, step)
    return nextU


def generateRK44Phi(f: Function) -> phiType:
    def phi(time: float, currentU: np.ndarray, step: float) -> np.ndarray:
        kappa1 = f(time, currentU)
        kappa2 = f(time + step / 2, currentU + step * kappa1 / 2)
        kappa3 = f(time + step / 2, currentU + step * kappa2 / 2)
        kappa4 = f(time + step, currentU + step * kappa3)
        return kappa1 / 6 + kappa2 / 3 + kappa3 / 3 + kappa4 / 6
    return phi


def generateimplicitEulerPhi(f: Function) -> phiType:
    def phi(time: float, currentU: np.ndarray, step: float) -> np.ndarray:
        def generateG(currentU: np.ndarray) -> Function:
            def g(time: float, nextU: np.ndarray):
                return np.array(nextU - step * f.getFs() - currentU)
            return g

        # TODO: Implement jacobian
        def generateJacobian(time: float, nextU: np.ndarray, currentU: np.ndarray) -> matrixType:
            g = generateG(currentU)
            jacobian = np.empty((len(nextU), len((nextU))))
            gs = g.getFs()
            for i in gs:
                jacobian[i] = np.gradient(gs[i](time, nextU), time)
            return jacobian

        def inverseJacobian(time: float, nextU: np.ndarray, currentU: np.ndarray) -> matrixType:
            return np.linalg.inv(generateJacobian(time, nextU, currentU))

        def newtonIteration(currentNextUAproximation: np.ndarray, previousNextUAproximation: np.ndarray):
            g = generateG(previousNextUAproximation)
            return currentNextUAproximation - inverseJacobian(time, currentNextUAproximation, previousNextUAproximation) * g(time + step, currentNextUAproximation)

        # TODO: Verify precision of Newton's method below
        nextU = previousNextUAproximation = currentNextUAproximation = currentU
        error = 0
        for iterations in range(MAX_NEWTON_ITERATIONS):
            previousNextUAproximation = currentNextUAproximation
            currentNextUAproximation = nextU
            if error < 1e-16:
                break
            nextU = newtonIteration(currentNextUAproximation,
                                    previousNextUAproximation)
            print("Newton iterations: ", iterations, ". Error: ",
                  "{:.3E}".format(Decimal(error)), "\r")
        print("\n")

        return nextU
    return phi
