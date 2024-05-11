import sys

from SciJo.LinearAlgebra.Types.Vector import Vector
from testing import assert_equal, assert_not_equal




fn test_getters_and_setters() raises:
    var vec = Vector(4)

    vec[0] = 5
    assert_equal(vec[0], 5)

    vec[1] = 4
    assert_equal(vec[1], 4)

    vec[2] = 3
    assert_equal(vec[2], 3)


fn test_safe_init() raises:
    var vec = Vector(4)

    assert_equal(vec[0], 0)
    assert_equal(vec[1], 0)
    assert_equal(vec[2], 0)
    assert_equal(vec[3], 0)


fn test_copy() raises:
    var vec1 = Vector(4)

    vec1[0] = 1
    vec1[1] = 2
    vec1[2] = 3
    vec1[3] = 4

    var vec2 = vec1

    assert_equal(vec1[0], 1)
    assert_equal(vec1[1], 2)
    assert_equal(vec1[2], 3)
    assert_equal(vec1[3], 4)

    assert_equal(vec1[0], 1)
    assert_equal(vec1[1], 2)
    assert_equal(vec1[2], 3)
    assert_equal(vec1[3], 4)

# fn test_addition() raises:
#     var mat1 = Matrix(2, 2)
#     var mat2 = Matrix(2, 2)

#     mat1[0, 0] = 1
#     mat1[0, 1] = 2
#     mat1[1, 0] = 3
#     mat1[1, 1] = 4

#     mat2[0, 0] = 1
#     mat2[0, 1] = 2
#     mat2[1, 0] = 3
#     mat2[1, 1] = 4

#     var mat_add = mat1 + mat2

#     assert_equal(mat_add[0, 0], 2)
#     assert_equal(mat_add[0, 1], 4)
#     assert_equal(mat_add[1, 0], 6)
#     assert_equal(mat_add[1, 1], 8)


# fn test_subtraction() raises:
#     var mat1 = Matrix(2, 2)
#     var mat2 = Matrix(2, 2)

#     mat1[0, 0] = 1
#     mat1[0, 1] = 2
#     mat1[1, 0] = 3
#     mat1[1, 1] = 4

#     mat2[0, 0] = 1
#     mat2[0, 1] = 2
#     mat2[1, 0] = 3
#     mat2[1, 1] = 4

#     var mat_sub = mat1 - mat2

#     assert_equal(mat_sub[0, 0], 0)
#     assert_equal(mat_sub[0, 1], 0)
#     assert_equal(mat_sub[1, 0], 0)
#     assert_equal(mat_sub[1, 1], 0)


# fn test_multiplication() raises:
#     var mat1 = Matrix(2, 2)
#     var mat2 = Matrix(2, 2)

#     mat1[0, 0] = 1
#     mat1[0, 1] = 2
#     mat1[1, 0] = 3
#     mat1[1, 1] = 4

#     mat2[0, 0] = 1
#     mat2[0, 1] = 2
#     mat2[1, 0] = 3
#     mat2[1, 1] = 4

#     var mat_mul = mat1 * mat2

#     assert_equal(mat_mul[0, 0], 1)
#     assert_equal(mat_mul[0, 1], 4)
#     assert_equal(mat_mul[1, 0], 9)
#     assert_equal(mat_mul[1, 1], 16)


# fn test_division() raises:
#     var mat1 = Matrix(2, 2)
#     var mat2 = Matrix(2, 2)

#     mat1[0, 0] = 1
#     mat1[0, 1] = 2
#     mat1[1, 0] = 3
#     mat1[1, 1] = 4

#     mat2[0, 0] = 1
#     mat2[0, 1] = 2
#     mat2[1, 0] = 3
#     mat2[1, 1] = 4

#     var mat_div = mat1 / mat2

#     assert_equal(mat_div[0, 0], 1)
#     assert_equal(mat_div[0, 1], 1)
#     assert_equal(mat_div[1, 0], 1)
#     assert_equal(mat_div[1, 1], 1)


# fn test_matrix_multiplication() raises:
#     var mat1 = Matrix(2, 2)
#     var mat2 = Matrix(2, 2)

#     mat1[0, 0] = 1
#     mat1[0, 1] = 2
#     mat1[1, 0] = 3
#     mat1[1, 1] = 4

#     mat2[0, 0] = 1
#     mat2[0, 1] = 2
#     mat2[1, 0] = 3
#     mat2[1, 1] = 4

#     var matmul = mat1 @ mat2

#     assert_equal(mat_div[0, 0], 7)
#     assert_equal(mat_div[0, 1], 10)
#     assert_equal(mat_div[1, 0], 15)
#     assert_equal(mat_div[1, 1], 22)

# fn main() raises:

#     var mat1 = Matrix(100, 100)
#     var mat2 = Matrix(100, 100)

#     mat1.rand()
#     mat2.rand()

#     var new_mat = mat1 @ mat2 

#     new_mat.tofile("matmul.mat")

     

