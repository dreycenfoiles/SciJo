from SciJo.src.GaussianProcess.kernels import RBF
from SciJo.src.LinearAlgebra.Types.Matrix import Matrix, eye
from SciJo.src.LinearAlgebra.Types.Vector import Vector
from SciJo.src.LinearAlgebra.LU import LU
from SciJo.src.LinearAlgebra.Cholesky import Cholesky
from python import Python
from tensor.tensor import Tensor


fn matrix_to_numpy[dtype: DType](A: Matrix[dtype]) raises -> PythonObject:
    var np = Python.import_module("numpy")

    var rows = A.rows
    var cols = A.cols

    var numpy_array = np.zeros((rows, cols), np.float64)

    for i in range(rows):
        for j in range(cols):
            numpy_array.itemset((i, j), A[i, j])

    return numpy_array


fn vector_to_numpy[dtype: DType](A: Vector[dtype]) raises -> PythonObject:
    var np = Python.import_module("numpy")

    var size = A.size

    var numpy_array = np.zeros(size, np.float64)

    for i in range(size):
        numpy_array.itemset(i, A[i])

    return numpy_array

fn linspace[dtype: DType](t0: Scalar[dtype], t1: Scalar[dtype], n: Int) -> List[Scalar[dtype]]:

    var spaced_list = List[Scalar[dtype]](capacity=n)
    var step = (t1 - t0)/n

    var total = t0

    for i in range(n):
        total += step
        spaced_list[i] = total

    return spaced_list

fn main() raises:

    var num_points = 200

    var x_points = linspace(-5., 5., num_points)

    var mat = Matrix(num_points, num_points)
    var u = Vector(num_points)
    var u2 = u

    var plt = Python.import_module("matplotlib.pyplot")
  

    var rbf = RBF(l=1)
    for i in range(num_points):
        for j in range(num_points):
            mat[i,j] = rbf(Float64(x_points[i]), x_points[j]) 

    var L = Matrix(Cholesky(mat + (eye(num_points)*1e-6)))


    var np = Python.import_module("numpy")
    var py_x_points = np.zeros(num_points)

    for i in range(num_points):
        py_x_points.itemset(i, x_points[i])    


    var numpy_array: PythonObject


    plt.figure()
    for i in range(10):
        u.randn()
        u2 = L @ u 
        numpy_array = vector_to_numpy(u2)
        plt.plot(py_x_points, numpy_array)

    plt.show()


    

    
