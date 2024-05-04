from buffer import Buffer
from memory import memcpy
from algorithm import vectorize
from random import rand, seed
from pathlib.path import Path


struct Vector[dtype: DType = DType.float64]:
    var data: DTypePointer[dtype]
    var buffer: Buffer[dtype]

    var size: Int

    alias simd_width: Int = simdwidthof[dtype]()

    fn __init__(inout self, size: Int):
        self.size = size
        self.data = DTypePointer[dtype].alloc(self.size)

        self.buffer = Buffer[dtype](self.data, self.size)
        self.buffer.fill(0)

    fn __del__(owned self):
        self.data.free()

    fn __copyinit__(inout self, other: Self):
        self.size = other.size
        self.data = DTypePointer[dtype].alloc(self.size)
        memcpy(self.data, other.data, self.size)

        self.buffer = Buffer[dtype](self.data, self.size)

    fn __moveinit__(inout self, owned other: Self):
        self.size = other.size

        self.data = other.data
        self.buffer = Buffer[dtype](self.data, self.size)

    fn fill_rand(inout self):
        seed()
        rand(self.data, self.size)

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

        return self.buffer[idx]

    fn __setitem__(inout self, owned idx: Int, val: Scalar[dtype]):
        self.buffer[idx] = val

    fn __eq__(self, other: Self) -> Bool:
        for i in range(self.size):
            if self[i] != other[i]:
                return False

        return False

    fn tofile(self, path: Path) raises:
        self.buffer.tofile(path)
