from memory import memset_zero, memcpy
from algorithm import vectorize, parallelize, elementwise
from buffer.list import DimList
import math
from random import rand, seed
from math import max
from buffer.buffer import NDBuffer
from pathlib.path import Path
from utils.index import Index


struct Matrix[dtype: DType = DType.float64](Stringable, CollectionElement):
    var data: DTypePointer[dtype]
    var buffer: NDBuffer[dtype, 2]

    var size: Int
    var rows: Int
    var cols: Int

    alias simd_width: Int = simdwidthof[dtype]()

    ################
    # Initializers #
    ################

    fn __init__(inout self, rows: Int, cols: Int):
        self.size = rows * cols
        self.rows = rows
        self.cols = cols

        self.data = DTypePointer[dtype].alloc(self.size)
        self.buffer = NDBuffer[dtype, 2](
            self.data, DimList(self.rows, self.cols)
        )

        self.buffer.fill(0)

    fn __del__(owned self):
        self.data.free()

    fn __copyinit__(inout self, other: Self):
        self.size = other.size
        self.rows = other.rows
        self.cols = other.cols

        self.data = DTypePointer[dtype].alloc(self.size)
        memcpy(self.data, other.data, self.size)

        self.buffer = NDBuffer[dtype, 2](self.data, DimList(self.rows, self.cols))

    fn __moveinit__(inout self, owned other: Self):
        self.size = other.size
        self.rows = other.rows
        self.cols = other.cols

        self.data = other.data
        self.buffer = NDBuffer[dtype, 2](self.data, DimList(self.rows, self.cols))

    fn fill_rand(inout self):
        seed()
        rand(self.data, self.size)

    ###########
    # Getters #
    ###########

    fn _adjust_index_(self, idx: Int, dim: Int) -> Int:
        if idx < 0:
            return dim + idx
        else:
            return idx

    fn _adjust_slice_(self, inout span: Slice, dim: Int):
        if span.start < 0:
            span.start += dim
        if not span._has_end():
            span.end = dim
        elif span.end < 0:
            span.end += dim
        if span.end > dim:
            span.end = dim
        if span.end < span.start:
            span.start = 0
            span.end = 0

    fn __getitem__(self, owned row: Int, owned col: Int) -> Scalar[dtype]:
        row = self._adjust_index_(row, self.rows)
        col = self._adjust_index_(col, self.cols)

        return self.buffer[row, col]

    fn __getitem__(
        self, owned row_slice: Slice, owned col_slice: Slice
    ) -> Self:
        self._adjust_slice_(row_slice, self.rows)
        self._adjust_slice_(col_slice, self.cols)

        var sliced_mat = Self(len(row_slice), len(col_slice))

        @parameter
        fn slice_column(idx_row_slice: Int):
            @parameter
            fn slice_row[simd_width: Int](idx_col_slice: Int) -> None:
                var row_idx = row_slice.start + idx_row_slice * row_slice.step
                var col_idx = col_slice.start + idx_col_slice * col_slice.step

                sliced_mat.buffer.store[width=simd_width](
                    Index(idx_row_slice, idx_col_slice),
                    self.buffer.load[width=simd_width](Index(row_idx, col_idx)),
                )

            vectorize[slice_row, self.simd_width](len(col_slice))

        parallelize[slice_column](len(row_slice), len(row_slice))

        return sliced_mat

    ###########
    # Setters #
    ###########

    fn __setitem__(
        inout self, owned row: Int, owned col: Int, val: Scalar[dtype]
    ):
        row = self._adjust_index_(row, self.rows)
        col = self._adjust_index_(col, self.cols)

        self.buffer[Index(row, col)] = val

    fn __setitem__(
        inout self, owned row_slice: Slice, owned col_slice: Slice, val: Self
    ):
        self._adjust_slice_(row_slice, self.rows)
        self._adjust_slice_(col_slice, self.cols)

        @parameter
        fn slice_column(idx_row_slice: Int):
            @parameter
            fn slice_row[simd_width: Int](idx_col_slice: Int) -> None:
                var row_idx = row_slice.start + idx_row_slice * row_slice.step
                var col_idx = col_slice.start + idx_col_slice * col_slice.step

                self.buffer.store[width=simd_width](
                    Index(idx_row_slice, idx_col_slice),
                    val.buffer.load[width=simd_width](Index(row_idx, col_idx)),
                )

            vectorize[slice_row, self.simd_width](len(col_slice))

        parallelize[slice_column](len(row_slice), len(row_slice))

    ########
    # Math #
    ########

    fn __neg__(self) -> Self:
        var new_mat = Self(self.rows, self.cols)

        @parameter
        fn wrapper[simd_width: Int, rank: Int](idx: StaticIntTuple[rank]):
            var loc = Index(idx[0], idx[1])

            new_mat.buffer.store[width=simd_width](
                loc,
                -1 * self.buffer.load[width=simd_width](loc),
            )

        elementwise[wrapper, 2](StaticIntTuple[2](self.rows, self.cols))
        return new_mat

    fn _binary_op_[
        func: fn[dtype: DType, width: Int] (
            SIMD[dtype, width], SIMD[dtype, width]
        ) -> SIMD[dtype, width]
    ](self, other: Self) -> Self:
        var new_mat = Self(self.rows, self.cols)

        @parameter
        fn wrapper[simd_width: Int, rank: Int](idx: StaticIntTuple[rank]):
            var loc = Index(idx[1], idx[2])

            new_mat.buffer.store[width=simd_width](
                loc,
                func(
                    self.buffer.load[width=simd_width](loc),
                    other.buffer.load[width=simd_width](loc),
                ),
            )

        elementwise[wrapper, 2](StaticIntTuple[2](self.rows, self.cols))
        return new_mat

    fn __add__(self, other: Self) -> Self:
        return self._binary_op_[math.add](other)

    fn __sub__(self, other: Self) -> Self:
        return self._binary_op_[math.sub](other)

    fn __mul__(self, other: Self) -> Self:
        return self._binary_op_[math.mul](other)

    fn __truediv__(self, other: Self) -> Self:
        return self._binary_op_[math.div](other)

    # TODO Make more efficient
    fn __matmul__(self, other: Self) -> Self:
        var new_mat = Self(self.rows, self.cols)

        for i in range(self.rows):
            for j in range(self.cols):
                for k in range(other.rows):
                    new_mat[i, j] += self[i, k] * other[k, j]

        return new_mat

    # TODO Make prettier printing
    fn __str__(self) -> String:
        var printStr: String = "\n"

        for i in range(self.rows):
            for j in range(self.cols):
                printStr += " " + str(self[i, j])
            printStr += "\n"

        return printStr

    fn __eq__(self, other: Self) -> Bool:
        for i in range(self.rows):
            for j in range(self.cols):
                if self[i, j] != other[i, j]:
                    return False

        return True

    fn tofile(self, path: Path) raises:
        self.buffer.tofile(path)
