from tensor import TensorShape
from tensor import Tensor
from math import sqrt, exp
from utils.index import Index
from python import Python
from testing.testing import Testable
from SciJo.LinearAlgebra.Types.Matrix import Matrix


struct Cholesky[dtype: DType = DType.float64]:
    var tensor: Tensor[dtype]
    var n: Int
    var sum: Scalar[dtype]

    fn __init__(inout self, A: Matrix[dtype]):
        self.sum = 0
        self.n = A.rows
        self.tensor = Tensor[dtype](self.n, self.n)

        for i in range(self.n):
            for j in range(i+1):
                self.sum = 0 
                for k in range(j):
                    self.sum += self[i,k] * self[j,k]
                if i == j:
                    self[i,j] = sqrt(A[i,j] - self.sum)
                else:
                    self[i,j] = 1/self[j,j] * (A[i,j] - self.sum)

    fn __getitem__(self, row: Int, col: Int) -> Scalar[dtype]:
        return self.tensor[row, col]

    fn __setitem__(inout self, row: Int, col: Int, val: Scalar[dtype]):
        self.tensor[Index(row, col)] = val
