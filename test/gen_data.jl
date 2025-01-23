include("../src/opt_problem.jl")
using .opt_problem, CUDA, ProgressMeter, Serialization, Enzyme


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
        a = rand(Float64, size(prob.d, 1))
        b = rand(Float64, size(prob.d, 1))
        g, _ = partial_dual_root(prob, a, b)
        z, n, mvp = pade_root(x -> partial_dual_constraint(prob, x, a, b)[1:3], g)
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
a = ones(Float64, size(domains, 1))
b = ones(Float64, size(domains, 1))
prob = OptProblem(n_volume, domains)
partial_dual(prob)

# g, _ = partial_dual_root(prob, a, b)
# Define the function to be differentiated
# function partial_dual_fn_wrapper(prob, l, g)
#     a = l[1:size(prob.d, 1)]
#     b = l[size(prob.d, 1)+1:end]
#     return partial_dual_fn(prob, a, b, g)
# end

# Differentiate the function
# g, _ = partial_dual_root(prob, a, b)
# autodiff_result = autodiff((l) -> partial_dual_fn_wrapper(prob, l, g), vcat(a, b))

# @show first(autodiff_result[1])
# zs, ns, mvps = stat_test(n_samples, n_volume, domains)
# serialize("data/zeros_"*string(n_volume)*".dat", zs)
# serialize("data/iters_"*string(n_volume)*".dat", ns)
# serialize("data/mvps_"*string(n_volume)*".dat", ns)