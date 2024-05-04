from python import Python
from testing import assert_equal

from SciJo.src.LinearAlgebra.LU import LU
from SciJo.src.LinearAlgebra.Types.Matrix import Matrix
from algorithm.swap import swap




fn test_lu() raises:

    var n = 3

    var alpha = Matrix(n,n)
    var beta = Matrix(n,n)
    
    var mat = Matrix(n,n)
    mat.fill_rand()

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

    var permutted_mat = alpha @ beta

    for i in range(n):
        for j in range(n):
            swap(permutted_mat[i,j], permutted_mat[lu.index[i],j])

fn main() raises:
    
    test_lu()

    # print(my_lu.lu)


