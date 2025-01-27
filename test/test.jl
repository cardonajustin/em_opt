include("../src/utils.jl")
include("../src/opt_problem.jl")
using .utils, .opt_problem, GilaElectromagnetics, CUDA, LinearAlgebra


n =  2
G = load_greens_operator((n, n, n), (1//32, 1//32, 1//32))
m =  size(G, 2)
X =  CUDA.Diagonal((rand(Float64) + 1e-3im * rand(Float64)) * CUDA.ones(ComplexF64, m))
Oquad = CUDA.Diagonal(CUDA.ones(ComplexF64, m))
Olin = CUDA.randn(ComplexF64, m)
Ei = CUDA.randn(ComplexF64, m)
d = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18], [19, 20, 21], [22, 23, 24]]

prob = OptProblem(G, X, Oquad, Olin, Ei, d)
@show dual(prob)