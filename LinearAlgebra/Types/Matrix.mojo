from tensor import Tensor
from random import rand
from memory.buffer import NDBuffer
from memory import memset_zero
from utils.index import Index
from algorithm import vectorize, parallelize, elementwise
from random import rand


struct Matrix[dtype: DType = DType.float32](CollectionElement):
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
        self.buffer = NDBuffer[dtype, 2](self.data, DimList(self.rows, self.cols))
        self.buffer.fill(0)

    fn __del__(owned self):
        self.data.free()

    fn __copyinit__(inout self, other: Self):
        self.size = other.size
        self.rows = other.rows
        self.cols = other.cols

        self.data = DTypePointer[dtype].alloc(self.size)
        self.buffer = NDBuffer[dtype, 2](self.data, DimList(self.rows, self.cols))

        @parameter
        fn slice_column(idx_row: Int):
            @parameter
            fn slice_row[simd_width: Int](idx_col: Int) -> None:
                self.buffer.simd_store[simd_width](
                    Index(idx_row, idx_col),
                    self.buffer.simd_load[simd_width](Index(idx_row, idx_col)),
                )

            vectorize[slice_row, self.simd_width](self.cols)

        parallelize[slice_column](self.rows, self.rows)

    fn __moveinit__(inout self, owned other: Self):
        self.size = other.size
        self.rows = other.rows
        self.cols = other.cols

        self.data = other.data
        self.buffer = NDBuffer[dtype, 2](self.data, DimList(self.rows, self.cols))

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
            span.start = dim + span.start
        if not span._has_end():
            span.end = dim
        elif span.end < 0:
            span.end = dim + span.end
        if span.end > dim:
            span.end = dim
        if span.end < span.start:
            span.start = 0
            span.end = 0

    fn __getitem__(self, row: Int, col: Int) -> SIMD[dtype, 1]:
        var new_row = self._adjust_index_(row, self.rows)
        var new_col = self._adjust_index_(col, self.cols)

        return self.buffer[new_row, new_col]

    fn __getitem__(self, owned row: Int, owned col_slice: Slice) -> Self:
        return self.__getitem__(Slice(row, row + 1), col_slice)

    fn __getitem__(self, owned row_slice: Slice, owned col: Int) -> Self:
        return self.__getitem__(row_slice, Slice(col, col + 1))

    fn __getitem__(self, owned row_slice: Slice, owned col_slice: Slice) -> Self:
        self._adjust_slice_(row_slice, self.rows)
        self._adjust_slice_(col_slice, self.cols)

        var sliced_mat = Self(len(row_slice), len(col_slice))

        @parameter
        fn slice_column(idx_row_slice: Int):
            @parameter
            fn slice_row[simd_width: Int](idx_col_slice: Int) -> None:
                var row_idx = row_slice.start + idx_row_slice * row_slice.step
                var col_idx = col_slice.start + idx_col_slice * col_slice.step

                sliced_mat.buffer.simd_store[simd_width](
                    Index(idx_row_slice, idx_col_slice),
                    self.buffer.simd_load[simd_width](Index(row_idx, col_idx)),
                )

            vectorize[slice_row, self.simd_width](len(col_slice))

        parallelize[slice_column](len(row_slice), len(row_slice))

        return sliced_mat

    ###########
    # Setters #
    ###########

    fn __setitem__(inout self, row_idx: Int, col_idx: Int, val: Scalar[dtype]):
        var new_row = self._adjust_index_(row_idx, self.rows)
        var new_col = self._adjust_index_(col_idx, self.cols)

        self.buffer[Index(new_row, new_col)] = val

    fn __setitem__(inout self, owned row: Int, owned col_slice: Slice, val: Self):
        return self.__setitem__(Slice(row, row + 1), col_slice, val)

    fn __setitem__(inout self, owned row_slice: Slice, owned col: Int, val: Self):
        return self.__setitem__(row_slice, Slice(col, col + 1), val)

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

                self.buffer.simd_store[simd_width](
                    Index(idx_row_slice, idx_col_slice),
                    val.buffer.simd_load[simd_width](Index(row_idx, col_idx)),
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
            new_mat.buffer.simd_store[simd_width](
                Index(idx[0], idx[1]),
                -1 * self.buffer.simd_load[simd_width](Index(idx[0], idx[1])),
            )

        elementwise[2, self.simd_width, wrapper](self.buffer.get_shape())
        return new_mat

    fn _binary_op_[
        func: fn[dtype: DType, width: Int] (
            SIMD[dtype, width], SIMD[dtype, width]
        ) -> SIMD[dtype, width]
    ](self, other: Self) -> Self:
        var new_mat = Self(self.rows, self.cols)

        @parameter
        fn wrapper[simd_width: Int, rank: Int](idx: StaticIntTuple[rank]):
            new_mat.buffer.simd_store[simd_width](
                Index(idx[0], idx[1]),
                func(
                    self.buffer.simd_load[simd_width](Index(idx[0], idx[1])),
                    other.buffer.simd_load[simd_width](Index(idx[0], idx[1])),
                )
            )

        elementwise[2, self.simd_width, wrapper](self.buffer.get_shape())
        return new_mat

    fn __add__(self, other: Self) -> Self:
        return self._binary_op_[math.add](other)
    
    fn __sub__(self, other: Self) -> Self: 
        return self._binary_op_[math.sub](other)

    fn __mul__(self, other: Self) -> Self: 
        return self._binary_op_[math.mul](other)

    fn __truediv__(self, other: Self) -> Self: 
        return self._binary_op_[math.div](other)
    
    # fn __matmul__(self, other: Self) -> Self: 

    #     for 
        

fn main():
    var mat1 = Matrix(4, 4)
    var mat2 = Matrix(4, 4)

    mat1[0, 0] = 2
    mat2[0, 0] = 3
    mat2[1, 3] = 1

    var mat3 = mat1 + mat2
    print(mat3.buffer)
