from tensor import TensorShape
from tensor import Tensor
from math import sqrt, exp 
from utils.index import Index
from python import Python
from random import randn_float64

alias type = DType.float64

fn cholesky(A: Tensor[type]) -> Tensor[type]:

    var sum: Float64 = 0.

    let size = A.shape()[0]

    var L = Tensor[type](size, size)

    for i in range(size):
        for j in range(i+1):
            sum = 0
            for k in range(j):
                sum += L[i,k] * L[j,k]

            if i == j:
                L[Index(i,j)] = sqrt(A[j][j] - sum)

            else:
                L[Index(i,j)] = 1 / L[j,j] * (A[i][j] - sum)
    
    return L

def LU(A):

    np = Python.import_module("numpy")

    A.shape


