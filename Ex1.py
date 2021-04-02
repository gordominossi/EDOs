import numpy as np
import common


A = np.matrix([[-2, -1, -1, -2],
               [1, -2, 2, -1],
               [-1, -2, -2, -1],
               [2, -1, 1, -2]])

f = A

phiRK44 = common.RK44Phi(f, )
