from typing import Callable
import numpy as np

MAX_NEWTON_ITERATIONS = 10
phiType = Callable[[float, np.ndarray, float], np.ndarray]
matrixType = np.ndarray


class Function:
    def __init__(self, fArray: np.ndarray):
        self.fArray = fArray

    def __call__(self, time: float, u: np.ndarray) -> np.ndarray:
        return np.array([self.fArray[i](time, u[i]) for i in len(self.fArray)])

    def getFs(self) -> np.ndarray:
        return self.fArray


def fixedPointIteration(time: float, currentU: np.ndarray, step: float, phi: phiType) -> np.ndarray:
    nextU = currentU + step * phi(time, currentU, step)
    return nextU


def RK44Phi(f: Function, time: float, currentU: np.ndarray, step: float) -> phiType:
    kappa1 = f(time, currentU)
    kappa2 = f(time + step / 2, currentU + step * kappa1 / 2)
    kappa3 = f(time + step / 2, currentU + step * kappa2 / 2)
    kappa4 = f(time + step, currentU + step * kappa3)
    return kappa1 / 6 + kappa2 / 3 + kappa3 / 3 + kappa4 / 6


def implicitEulerPhi(f: Function, time: float, currentU: np.ndarray, step: float) -> phiType:
    def g(time: float, nextU: np.ndarray, currentU) -> np.ndarray:
        return np.array(nextU - step * f.getFs() - currentU)

    # TODO: Implement jacobian
    def jacobian(time: float, nextU: np.ndarray, currentU: np.ndarray) -> matrixType:
        return np.full((len(nextU), len(nextU)), g(time, nextU, currentU))

    def inverseJacobian(time: float, nextU: np.ndarray, currentU: np.ndarray) -> matrixType:
        return np.linalg.inv(jacobian(time, nextU, currentU))

    def newtonIteration(currentNextUAproximation: np.ndarray, previousNextUAproximation: np.ndarray):
        return currentNextUAproximation - inverseJacobian(time, currentNextUAproximation, previousNextUAproximation) * g(time, currentNextUAproximation, previousNextUAproximation)

    # TODO: Verify precision of Newton's method below
    nextU = previousNextUAproximation = currentNextUAproximation = currentU
    error = 0
    for dummyIterationCounter in range(MAX_NEWTON_ITERATIONS):
        previousNextUAproximation = currentNextUAproximation
        currentNextUAproximation = nextU
        if error < 1e-16:
            break
        nextU = newtonIteration(currentNextUAproximation,
                                previousNextUAproximation)

    return nextU
