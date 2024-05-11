from python import Python
from testing import assert_equal, assert_almost_equal

from SciJo.LinearAlgebra.LU import LU
from SciJo.LinearAlgebra.Types.Matrix import Matrix
from SciJo.LinearAlgebra.Types.Vector import Vector
from algorithm.swap import swap
from tensor.tensor import Tensor



fn test_lu() raises:

    var n = 100

    var alpha = Matrix(n,n)
    var beta = Matrix(n,n)
    
    var mat = Matrix(n,n)
    mat.rand()

    var lu = LU(mat)

    for i in range(n):
        for j in range(n):
            if j < i:
                alpha[i,j] = lu[i,j]
            elif j > i:
                beta[i,j] = lu[i,j] 
            else:
                beta[i,j] = lu[i,j]
                alpha[i,j] = 1

    var permuted_mat = alpha @ beta
    var depermuted_mat = Matrix(n,n)

    for i in range(n):
        for j in range(n):
            depermuted_mat[lu.index[i],j] = permuted_mat[i,j]
    
    assert_equal(depermuted_mat, mat)


fn test_solve() raises:

    # Do not use n = 4. That segfaults for some reason 
    var n = 10
    
    var mat = Matrix(n,n)
    mat.rand()

    var vec = Vector(n)
    vec.rand()

    var lu = LU(mat)

    var sol = lu.solve(vec)

    var vec_guess = mat @ sol 

    assert_equal(vec_guess, vec)

