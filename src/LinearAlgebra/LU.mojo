from SciJo.src.LinearAlgebra.Types.Matrix import Matrix
from algorithm.swap import swap
from memory.memory import memcpy
from .Types.Vector import Vector

struct LU[dtype: DType]:
    alias TINY = 1e-40

    var n: Int

    var vv: DTypePointer[dtype]  # TODO Give a better name
    var index: List[Int]
    var d: Int  # Stores the sign / #TODO Give a better name

    var data: DTypePointer[dtype]

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

        self.data = DTypePointer[dtype].alloc(self.n * self.n)

        self.vv = DTypePointer[dtype].alloc(self.n)
        # self.index = 
        
        self.index = List[Int](capacity=self.n)

        for i in range(0,self.n):
            self.index[i] = i 
        
        memcpy(self.data, mat.data, self.n*self.n)

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
                self.tmp = self.vv[i]*abs(self[i, k])
                if self.tmp > self.big:
                    self.big = self.tmp
                    self.imax = i

            if k != self.imax:
                for j in range(0, self.n):
                    swap(self[k, j], self[self.imax, j])
                self.d = -self.d
                self.index[self.imax] = self.index[k]

            if self[k, k] == 0:
                self[k, k] = self.TINY

            for i in range(k + 1, self.n):
                self[i, k] /= self[k, k]
                self.tmp = self[i, k]
                for j in range(k + 1, self.n):
                    self[i, j] -= self.tmp * self[k, j]

    fn __del__(owned self):
        self.data.free()

    fn __getitem__(self, row: Int, col: Int) -> Scalar[dtype]:
        var offset = col + row * self.n
        return self.data[offset]

    fn __setitem__(inout self, row: Int, col: Int, val: Scalar[dtype]):
        var offset = col + row * self.n 
        self.data[offset] = val 

    fn __str__(self) -> String:

        var printStr: String = "\n"
        
        for i in range(self.n):
            for j in range(self.n):
                printStr += " " + str(self[i,j])
            printStr += "\n"

        return printStr


    fn solve(self, b: Vector[dtype]):

        
