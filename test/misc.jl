include("../src/opt_problem.jl")
include("../src/utils.jl")
using .opt_problem, .utils, CUDA, LinearAlgebra, ReverseDiff

n = 2
G = load_greens_operator((n, n, n), (1//32, 1//32, 1//32))
m = size(G, 2)

X = CUDA.Diagonal(ComplexF64(rand(Float64) - 0.5 + 1e-3im * rand(Float64)) .* CUDA.ones(ComplexF64, m))
Q_mat = CUDA.Diagonal(ComplexF64.(CUDA.ones(Float64, m)))    
Q = x-> Q_mat * x
S = CUDA.rand(ComplexF64, m) .- ComplexF64(0.5 + 0.5im)
d = [[1, 2, 3], [5, 7, 8], [1, 4, 9, 10], [4, 5]]
p = OptProblem(G, X, Q, S, d)

g, _ = partial_dual_root(p, ones(Float64, 8))

@show partial_dual(p)