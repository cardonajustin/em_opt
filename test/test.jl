include("../src/utils.jl")
include("../src/opt_problem.jl")
using .utils, .opt_problem, LinearAlgebra, JLD2, Plots

n =  2
m =  3 * n^3
Ol = randn(ComplexF64, m)
Oq = Diagonal(zeros(ComplexF64, m))
d = [[i, i+1, i+2] for i in 1:3:m]
d = vcat(d, [[i for i in 1:m]])
# d = [[i for i in 1:m]]
Ei = randn(ComplexF64, m)
# G = greens_matrix((n, n, n), (1//32, 1//32, 1//32), "data/greens_matrix_cpu_$n.jld2")
G = jldopen("data/greens_matrix_cpu_2.jld2")["result"]
X =  rand(Float64) + 1e-3im * rand(Float64)
prob = OptProblem(Ol, Oq, d, Ei, G, X)

P = randn(ComplexF64, m)
l = zeros(Float64, 4 * length(d))
l[2 * length(d)] = 1.0
g = 0.0
@show dual(prob, x0=l)
# xs = LinRange(0.006, 0.008, 100)
# ys = LinRange(0.95, 1.05, 100)
# f = (x,y)-> norm(dual_grad(prob, [x, y, 0, 0], 0.0))
# plot(xs, ys, f, st=:surface)

# xs = LinRange(1, 1.1, 10)
# ys = f.(xs)
# plot(xs, ys, label="dual_grad", xlabel="x", ylabel="f(x)", title="dual_grad")
# savefig("data/dual_grad.png")

# x=range(-2,stop=2,length=100)
# y=range(sqrt(2),stop=2,length=100)
# f(x,y) = x*y-x-y+1
# plot(x,y,f,st=:surface,camera=(-30,30))

