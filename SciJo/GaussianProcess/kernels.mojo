from math import exp, sin

alias pi = 3.14159265359

# TODO make more general when traits support parameters

# trait Kernel:
#     fn __call__(self, x: Float64, y: Float64) -> Float64:
#         pass


struct RBF[dtype: DType = DType.float64]:

    alias simd_width = simdwidthof[dtype]()

    var l: Scalar[dtype]  # Length scale parameter

    fn __init__(inout self, l: Float64):
        self.l = l

    fn __call__(self, x: Scalar[dtype], y: Scalar[dtype]) -> Scalar[dtype]:
        return exp(-((x - y) ** 2) / (2 * self.l ** 2))

    fn __call__(self, x: SIMD[dtype, self.simd_width], y: SIMD[dtype, self.simd_width]) -> SIMD[dtype, self.simd_width]:
        return exp(-((x - y) ** 2) / (2 * self.l ** 2))



# struct ExpSineSquared(Kernel):

#     var p: Float64 # Periodicity parameter
#     var l: Float64 # Length scale parameter

#     fn __init__(inout self, p: Float64, l: Float64):
#         self.p = p 
#         self.l = l 

#     fn __call__(self, x: Float64, y: Float64) -> Float64:

#         return exp(-(2 * ))


    