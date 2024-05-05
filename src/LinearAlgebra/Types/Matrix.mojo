from algorithm import vectorize, parallelize, elementwise
from math import isclose
from pathlib.path import Path
from utils.index import Index
from SciJo.src.LinearAlgebra.Types.Vector import Vector
from testing.testing import Testable
from tensor.tensor import Tensor
from random.random import rand


struct Matrix[dtype: DType = DType.float64](
    Stringable, CollectionElement, Testable
):
    var tensor: Tensor[dtype]

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

        self.tensor = Tensor[dtype](self.rows, self.cols)

    fn __init__(inout self, tensor: Tensor[dtype]):
        self.rows = tensor.shape()[0]
        self.cols = tensor.shape()[1]
        self.size = self.rows * self.cols

        self.tensor = tensor

    fn __copyinit__(inout self, other: Self):
        self.size = other.size
        self.rows = other.rows
        self.cols = other.cols

        self.tensor = other.tensor

    fn __moveinit__(inout self, owned other: Self):
        self.size = other.size
        self.rows = other.rows
        self.cols = other.cols

        self.tensor = other.tensor^

    fn rand(inout self):
        rand[dtype](self.tensor.data(), self.size)

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

        return self.tensor[row, col]

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

                sliced_mat.store(
                    idx_row_slice,
                    idx_col_slice,
                    self.load[simd_width](row_idx, col_idx),
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

        self.tensor[Index(row, col)] = val

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

                self.store[simd_width](
                    idx_row_slice,
                    idx_col_slice,
                    val.load[simd_width](row_idx, col_idx),
                )

            vectorize[slice_row, self.simd_width](len(col_slice))

        parallelize[slice_column](len(row_slice), len(row_slice))

    ########
    # Math #
    ########

    fn __neg__(self) -> Self:
        var new_mat = Self(self.rows, self.cols)
        new_mat.tensor = self.tensor * -1.0
        return new_mat

    fn __add__(self, other: Self) raises -> Self:
        return Self(self.tensor + other.tensor)

    fn __sub__(self, other: Self) raises -> Self:
        return Self(self.tensor - other.tensor)

    fn __mul__(self, other: Self) raises -> Self:
        return Self(self.tensor * other.tensor)

    fn __truediv__(self, other: Self) raises -> Self:
        return Self(self.tensor / other.tensor)

    # TODO Make more efficient
    fn __matmul__(self, other: Self) raises -> Self:
        var new_mat = Self(self.rows, self.cols)

        for i in range(self.rows):
            for j in range(self.cols):
                for k in range(other.rows):
                    new_mat[i, j] += self[i, k] * other[k, j]

        return new_mat

    fn __matmul__(self, other: Vector[dtype]) raises -> Vector[dtype]:
        var new_vec = Vector[dtype](self.rows)

        for i in range(self.rows):
            for j in range(self.cols):
                new_vec[i] += self[i, j] * other[j]

        return new_vec

    # TODO Make prettier printing
    fn __str__(self) -> String:
        var printStr: String = "\n"

        for i in range(self.rows):
            for j in range(self.cols):
                printStr += str(self[i, j]) + " "
            printStr += "\n"

        return printStr

    fn __eq__(self, other: Self) -> Bool:
        for i in range(self.rows):
            for j in range(self.cols):
                if not isclose(self[i, j], other[i, j]):
                    return False
        return True

    fn __ne__(self, other: Self) -> Bool:
        return not self == other

    fn T(self) -> Self:
        
        var new_mat = Matrix[dtype](self.cols,self.rows)

        for i in range(self.rows):
            for j in range(self.cols):
                new_mat[j,i] = self[i,j]

        return new_mat

    fn load[width: Int](self, row: Int, col: Int) -> SIMD[dtype, width]:
        return self.tensor.load[width=width](row, col)

    fn store[
        width: Int
    ](inout self, row: Int, col: Int, val: SIMD[dtype, width]):
        self.tensor.store[width=width](Index(row, col), val)

    fn tofile(self, path: Path) raises:
        self.tensor.tofile(path)

    fn fromfile(inout self, path: Path) raises:
        self.tensor = self.tensor.fromfile(path)
