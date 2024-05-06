# Overview 

This is intended to be a collection of common algorithms used in data science/scientific computing written in [Mojo](https://github.com/modularml/mojo). Inspired by [SciPy](https://github.com/scipy/scipy) 

Because Mojo does not have a package manager at the time of writing, I am throwing all the algorithms developed into this single repository. As Mojo packaging improves I will probably want to spine off some of the algorithms into more coherent repositories.

# Design

Because Mojo is a very new language, a lot has to be written from scratch like a `Matrix` and `Vector` type. As Mojo's ecosystem developes, I hope I can make use of other packages to reduce the amount of work. [NuMojo](https://github.com/MadAlex1997/NuMojo) is a similar project in this space.

# Roadmap 

I am inspired by SciPy and would like to include most of its functionalities but, at this time, I am not interested in exactly replicating its functionality or API. These are the broad areas that I am interested in adding. If you have other interests

## Near-term
- Linear systems
    - [ ] LU decomposition
    - [ ] Cholesky decomposition
    - [ ] QR decomposition
    - [ ] SVD
    - [ ] Sparse matricies
    - [ ] Special matrix patterns and solvers
    - [ ] Congugate-gradient
    - [ ] GMRES
- RNG
    - [ ] Multivariate Gaussian
    - [ ] Other distributions as needed (waiting on traits imporovements to define interfaces) 
- Modeling methods
    - [ ] Linear least squares
    - [ ] Nonlinear least squares
    - [ ] Gaussian process regression
- [ ] More testing
- [ ] Benchmarking
## Mid-term
- [ ] GPU algorithms (when available)
- [ ] Traits to define API (need parametric traits and default methods because this can be really useful)
- [ ] Actual documentation 
- Eigensystems
    - [ ] Jacobi transformation
    - [ ] QR method
- Fourier transforms
    - [ ] FFT's (both complex and real)
    - [ ] Cosine and sine transform
    - [ ] Multi-dimensional FFT
- Integration
    - [ ] Trapezoid and Rhomberg
    - [ ] Adaptive quadrature
    - [ ] Multi-dimensional integrals
- Optimization
    - [ ] Newton and quasi-Newton
    - [ ] Conjugant gradient
    - [ ] Simplex method
    - [ ] Powell's method
    - [ ] Simulated annealing
## Long-term
- Differential equations (less inspiration from SciPy and more from [DifferentialEquations.jl](https://github.com/SciML/DifferentialEquations.jl))
    - [ ] Runge-Kutta 4 and Dormand-Prince (Maybe Tsitouras?)
    - [ ] Backward Euler (and other stiff methods)  
    - [ ] Stochastic methods?
- Interpolation
    - [ ] BSplines
    - [ ] Interpolation on regular and irregular grids
    - [ ] Chebyshev polynomials
- Nonlinear equations and root finding
    - [ ] Newton-Raphson
    - [ ] Bisect
    - [ ] Ridder
    - [ ] Brent

 ## LONG-term 
 - Smart solvers
     - [ ] Pattern detection for linear solvers
     - [ ] Stiff and non-stiff switching in ODE's
 - Automatic differentiation compatability
     -  I do not have the skills to create a general autodiff library but I imagine someone in the Mojo community will have put one together by the time we get here. I would like to make it possible for the above algorithms to accept an AD type (whatever that may look like)
 - Better than "numerical analysis 101" algorithms

This roadmap is liable to change depending on the feasibility of algorithm implementation. 

# Contributing

I would be happy to have others take part in this project. The roadmap is obvious quite large and I doubt I would be able to do it by myself. Even, if an algorithm isn't on the roadmap, I'll gladly look over a contribution. I'd be especially appreciative of people who could make the existing algorithms more efficient. I am not not an expert in this field and most of my implementations are naive and probably have some inefficiencies. 

If you don't have a lot of experience with numerical methods, you're in good company! I'm pretty new to the field and getting a lot of my information from [Numerical Recipes](https://numerical.recipes/book.html) as a starting resource.

# Disclaimer

I am not a computer scientist or mathematician. I am simply (non-software) engineer with an interest in numerical methods and scientific computing. I would not recommend using any of this code for important tasks until it has been rigorously tested.
