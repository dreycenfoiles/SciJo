import numpy as np
from scipy.linalg import lu_factor


def LU(A):

    n = A.shape[0]

    LU = np.empty_like(A)

    for j in range(n):

        for i in range(j):

            val = 0

            for k in range(i):
                val += LU[i, k] * LU[k, j]

            LU[i, j] = A[i, j] - val

        for i in range(j + 1, n):

            val = 0

            for k in range(j):
                val += LU[i, k] * LU[k, j]

            LU[i, j] = 1 / LU[j, j] * (A[i, j] - val)

    return LU


A = np.random.random((10, 10))

lu_factor(A)
