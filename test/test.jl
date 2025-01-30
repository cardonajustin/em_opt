include("../src/utils.jl")
include("../src/opt_problem.jl")
using .utils, .opt_problem, GilaElectromagnetics, CUDA, LinearAlgebra

n =  2
m =  3 * n^3
Ol = CUDA.randn(ComplexF64, m)
Oq = CUDA.Diagonal(CUDA.ones(ComplexF64, m))
d = [[i for i in 1:m]]
Ei = CUDA.randn(ComplexF64, m)
G = load_greens_operator((n, n, n), (1//32, 1//32, 1//32))
X =  CUDA.Diagonal((rand(Float64) + 1e-3im * rand(Float64)) * CUDA.ones(ComplexF64, m))
prob = OptProblem(Ol, Oq, d, Ei, G, X)

P = CUDA.randn(ComplexF64, m)
l = rand(4 * length(d))
g = rand()
@show dual(prob)