from testing import assert_equal, assert_almost_equal

from SciJo.LinearAlgebra.Cholesky import Cholesky
from SciJo.LinearAlgebra.Types.Matrix import Matrix
from SciJo.LinearAlgebra.Types.Vector import Vector
from 

from algorithm.swap import swap
from tensor.tensor import Tensor


fn test_cholesky() raises:

    # TODO make better tests
    var n = 2
    var mat1 = Matrix(n, n)

    mat1[0,0] = 2
    mat1[1,1] = 1

    var mat2 = Matrix(n, n)

    var L = Cholesky(mat1)
    mat2.tensor = L.tensor

    assert_equal(mat1, mat2 @ mat2.T())

