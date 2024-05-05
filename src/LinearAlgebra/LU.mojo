from SciJo.src.LinearAlgebra.Types.Matrix import Matrix
from algorithm.swap import swap
from .Types.Vector import Vector
from utils.index import Index
from tensor.tensor import Tensor


struct LU[dtype: DType]:
    alias TINY = 1e-40

    var n: Int

    var vv: List[Scalar[dtype]]  # TODO Give a better name
    var index: List[Int]
    var d: Int  # Stores the sign / #TODO Give a better name

    var tensor: Tensor[dtype]

    var big: Scalar[dtype]
    var tmp: Scalar[dtype]

    var imax: Int

    fn __init__(inout self, mat: Matrix[dtype]):
        # TODO do out-of-place and in-place operations in same code

        self.big = 0
        self.tmp = 0
        self.imax = 0
        self.d = 1

        self.n = mat.cols

        self.vv = List[Scalar[dtype]](capacity=self.n)

        self.index = List[Int](capacity=self.n)

        for i in range(self.n + 1):
            self.index[i] = i

        self.tensor = mat.tensor

        # Normalize each row

        for i in range(0, self.n):
            self.big = 0

            for j in range(0, self.n):
                self.tmp = abs(self[i, j])
                if self.tmp > self.big:
                    self.big = self.tmp

            self.vv[i] = 1 / self.big

        for k in range(0, self.n):
            self.big = 0
            self.imax = k

            for i in range(k, self.n):
                self.tmp = self.vv[i] * abs(self[i, k])
                if self.tmp > self.big:
                    self.big = self.tmp
                    self.imax = i

            if k != self.imax:
                for j in range(0, self.n):
                    swap(self[k, j], self[self.imax, j])
                self.d = -self.d
                self.vv[self.imax] = self.vv[k]
            self.index[k] = self.imax

            if self[k, k] == 0:
                self[k, k] = self.TINY

            for i in range(k + 1, self.n):
                self[i, k] /= self[k, k]
                self.tmp = self[i, k]
                for j in range(k + 1, self.n):
                    self[i, j] -= self.tmp * self[k, j]

    fn __copyinit__(inout self, other: Self):
        self.tensor = other.tensor
        self.n = other.n
        self.vv = other.vv
        self.index = other.index
        self.d = other.d
        self.big = other.big
        self.tmp = other.tmp
        self.imax = other.imax

    fn __getitem__(self, row: Int, col: Int) -> Scalar[dtype]:
        return self.tensor[row, col]

    fn __setitem__(inout self, row: Int, col: Int, val: Scalar[dtype]):
        self.tensor[Index(row, col)] = val

    fn __str__(self) -> String:
        var printStr: String = ""

        for i in range(self.n):
            printStr += "\n"
            for j in range(self.n):
                printStr += str(self[i, j]) + " "

        printStr += "\n"

        return printStr

    fn solve(self, b: Vector[dtype]) raises -> Vector[dtype]:
        var solution = b

        for i in reversed(range(self.n)):
            for j in range(self.n):
                swap(solution[i], solution[self.index[i]])

        var total: Scalar[dtype] = 0

        # lower triangular

        for i in range(self.n):
            total = 0
            for j in range(i):
                total += self[i, j] * solution[j]
            solution[i] = b[i] - total

        # upper triangular
        for i in reversed(range(self.n)):
            total = 0
            for j in range(i + 1, self.n):
                total += self[i, j] * solution[j]
            solution[i] = (solution[i] - total) / self[i, i]

        return solution
