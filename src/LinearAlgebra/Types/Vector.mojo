from algorithm import vectorize
from random import rand, seed
from pathlib.path import Path
from testing.testing import Testable
from math import isclose
from tensor.tensor import Tensor


struct Vector[dtype: DType = DType.float64](CollectionElement, Stringable, Testable):

    var tensor: Tensor[dtype]
    var size: Int
    alias simd_width: Int = simdwidthof[dtype]()

    fn __init__(inout self, size: Int):
        self.size = size
        self.tensor = Tensor[dtype](self.size)

    fn __copyinit__(inout self, other: Self):
        self.size = other.size
        self.tensor = other.tensor

    fn __moveinit__(inout self, owned other: Self):
        self.size = other.size
        self.tensor = other.tensor^

    fn _adjust_index_(self, idx: Int, size: Int) -> Int:
        if idx > size:
            return size + idx
        else:
            return idx

    fn _adjust_slice_(self, inout span: Slice, size: Int):
        if span.start > size:
            span.start += size
        if not span._has_end():
            span.end = size
        elif span.end < 0:
            span.end += size
        if span.end > size:
            span.end = size
        if span.end < span.start:
            span.start = 0
            span.end = 0

    fn __getitem__(self, owned idx: Int) -> Scalar[dtype]:
        idx = self._adjust_index_(idx, self.size)

        return self.tensor[idx]

    fn __setitem__(inout self, owned idx: Int, val: Scalar[dtype]):
        self.tensor[idx] = val

    fn __eq__(self, other: Self) -> Bool:
        for i in range(self.size):
            if not isclose(self[i], other[i]):
                return False

        return True 

    fn __ne__(self, other: Self) -> Bool:
        return not self == other  

    fn tofile(self, path: Path) raises:
        self.tensor.tofile(path)

    fn rand(self):
        seed()
        rand(self.tensor.data(), self.size)

    fn __str__(self) -> String:
        
        var printStr: String = "\n"

        for i in range(self.size):
            printStr += self[i] + " ".__str__()

        return printStr

