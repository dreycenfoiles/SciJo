

trait Matrix(CollectionElement):

    fn __getitem__(inout self, row_idx: Int, col_idx: Int):

    fn __add__(self, other: Self) -> Self:
        ...

    fn __sub__(self, other: Self) -> Self: 
        ...
