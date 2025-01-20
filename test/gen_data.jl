include("../src/opt_problem.jl")
using .opt_problem, CUDA, ProgressMeter, Serialization


function stat_test(n_samples::Int, n_volume::Int, d::Vector{Vector{Int64}})
    zs = zeros(ComplexF64, n_samples)
    ns = zeros(Int, n_samples)
    mvps = zeros(Int, n_samples)
    prob = OptProblem(n_volume, d)
    for i in 1:n_samples
		println("Sample " * string(i))
        m = size(prob.G, 2)
        prob.X = CUDA.Diagonal(ComplexF64(rand(Float64) - 0.5 + 1e-3im * rand(Float64)) .* CUDA.ones(ComplexF64, m))   
        prob.S = CUDA.rand(ComplexF64, m) .- ComplexF64(0.5 + 0.5im)
        a = rand(Float64, size(prob.G, 2))
        b = rand(Float64, size(prob.G, 2))
        g, _ = partial_dual_root(prob, a, b)
        z, n, mvp = pade_root(x-> partial_dual_constraint(prob, x, a, b)[1:3], g)
        zs[i] = z
        ns[i] = n
        mvps[i] = mvp
        @show z, n, mvp
    end
    return zs, ns, mvps
end


n_samples = 100
n_volume = 2
domains = [[1, 2, 3], [5, 7, 8], [1, 4, 9, 10], [4, 5]]

# a = ones(Float64, size(domains, 1))
# b = ones(Float64, size(domains, 1))
# p = OptProblem(n_volume, domains)
# g = partial_dual_root(p, a, b)[1]
# s = partial_dual(p, a, b, g)
# @show s

zs, ns, mvps = stat_test(n_samples, n_volume, domains)
serialize("data/zeros_"*string(n_volume)*"_"*string(dx)*".dat", zs)
serialize("data/iters_"*string(n_volume)*"_"*string(dx)*".dat", ns)
serialize("data/mvps_"*string(n_volume)*"_"*string(dx)*".dat", ns)