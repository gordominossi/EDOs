from typing import Callable
import numpy as np
from decimal import Decimal

MAX_NEWTON_ITERATIONS = 10
phiType = Callable[[float, np.ndarray, float], np.ndarray]
fType = Callable[[float, np.ndarray], np.ndarray]
matrixType = np.ndarray


class Function:
    def __init__(self, fMatrix: np.ndarray):
        self.fMatrix = fMatrix

    def __call__(self, time: float, u: np.ndarray) -> np.ndarray:
        return np.matmul(self.fMatrix, u)

    def getFs(self) -> np.ndarray:
        return self.fMatrix


def solveEDO(u0: np.ndarray, phi: phiType, period: np.ndarray, discretization: int):
    U = np.array(u0)
    step = (period[1] - period[0]) / discretization
    currentU = u0
    nextU = currentU
    for iterations in range(discretization):
        nextU = iteration(
            period[0] + step * iterations, currentU, step, phi)
        np.append(U, nextU)
    return U


def iteration(time: float, currentU: np.ndarray, step: float, phi: phiType) -> np.ndarray:
    nextU = currentU + step * phi(time, currentU, step)
    return nextU


def RK44Phi(f: Function, time: float, currentU: np.ndarray, step: float) -> phiType:
    kappa1 = f(time, currentU)
    kappa2 = f(time + step / 2, currentU + step * kappa1 / 2)
    kappa3 = f(time + step / 2, currentU + step * kappa2 / 2)
    kappa4 = f(time + step, currentU + step * kappa3)
    return kappa1 / 6 + kappa2 / 3 + kappa3 / 3 + kappa4 / 6


def implicitEulerPhi(f: Function, time: float, currentU: np.ndarray, step: float) -> phiType:
    def generateG(time: float, nextU: np.ndarray, currentU: np.ndarray) -> Function:
        return Function(np.array(nextU - step * f.getFs() - currentU))

    # TODO: Implement jacobian
    def generateJacobian(time: float, nextU: np.ndarray, currentU: np.ndarray) -> matrixType:
        g = generateG(time, nextU, currentU)
        jacobian = np.empty((len(nextU), len((nextU))))
        gs = g.getFs()
        for i in gs:
            jacobian[i] = np.gradient(gs[i](time, nextU), time)
        return jacobian

    def inverseJacobian(time: float, nextU: np.ndarray, currentU: np.ndarray) -> matrixType:
        return np.linalg.inv(generateJacobian(time, nextU, currentU))

    def newtonIteration(currentNextUAproximation: np.ndarray, previousNextUAproximation: np.ndarray):
        g = generateG(time, currentNextUAproximation,
                      previousNextUAproximation)
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
