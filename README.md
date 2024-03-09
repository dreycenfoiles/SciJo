# Overview 

This is an attempt to replicate many of [SciPy's](https://github.com/scipy/scipy) functions using the Mojo programmign language. 

At this time, I am throwing all the algorithms developed into this single repository. Depending on how this repo and Mojo packing conventions develop, I may want to split it up into several repositories.

# Roadmap 

I am inspired by SciPy and would like to include most of its functionalities but I am (at this time) neither focused on porting over everything nor recreating its API. Given that, these are the broad areas I would like to include: 

- Solutions to linear systems
    - LU decomposition
    - QR decomposition
    - SVD 
- Eigensystems
    - Jacobi transformation
    - QR method
- Fourier transforms
    - FFT's (both complex and real)
    - Cosine and sine transform
    - Multi-dimensional FFT
- Differential equations (less inspiration from SciPy and more from [DifferentialEquations.jl](https://github.com/SciML/DifferentialEquations.jl))
    - Runge-Kutta 4 and Dormand-Prince
        - Maybe Tsitouras?
    - Backward Euler (and other stiff methods)  
    - Stochastic methods?
- Interpolation
    - BSplines
    - Interpolation on regular and irregular grids
- Integration
    - Trapezoid and Rhomberg
    - Adaptive quadrature
    - Multi-dimensional integrals
- Special functions ([already being handled](https://github.com/leandrolcampos/specials/))
- Optimization
    - Newton and quasi-Newton
    - Conjugant gradient
    - Simplex method
    - Powell's method
    - Simulated annealing
- Nonlinear equations and root finding
    - Newton-Raphson
    - Bisect
    - Ridder
    - Brent 
- Modeling methods
    - Linear least squares
    - Nonlinear least squares
    - Gaussian process regression

This roadmap is liable to change depending on the feasibility of algorithm implementation. 


# Contributing

I would be happy to have others take part in this project. Even, if an algorithm isn't on the roadmap, I'll gladly look over a contribution. I'd be especially appreciative of people who could make the existing algorithms more efficient. As stated below, I am not not an expert in this field and most of my implementations are naive and likely introducing inefficiencies. 

If you don't have a lot of experience with numerical methods, you're in good company! I'm pretty new to the field and getting a lot of my information from [Numerical Recipes](https://numerical.recipes/book.html) as a starting resource.

# Disclaimer

I am not a computer scientist or mathematician. I am simply (non-software) engineer with an interest in numerical methods and scientific computing. I would not recommend using any of this code for important tasks until it has been rigorously tested.