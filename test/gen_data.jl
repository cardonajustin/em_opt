include("../src/opt_problem.jl")
using .opt_problem, CUDA, ProgressMeter, Serialization


function zeros_perturbed(n_perturb::Int, n_volume::Int, dx::Float64=1e-2)
    zs = zeros(ComplexF64, n_perturb)
    ns = zeros(Int, n_perturb)
    C = OptProblem(n_volume)
	a = CUDA.Diagonal(CUDA.randn(ComplexF64, size(C.G, 2)))
	b = CUDA.Diagonal(CUDA.randn(ComplexF64, size(C.G, 2)))
	g, _ = partial_dual_root(C, a, b)
	println("initial guess done")
    for i in 1:n_perturb
		println("\t" * string(i) * " perturbations")
        a += CUDA.Diagonal(dx .* CUDA.randn(ComplexF64, size(C.G, 2)))
        b += CUDA.Diagonal(dx .* CUDA.randn(ComplexF64, size(C.G, 2)))
		result = pade_root(x-> partial_dual_constraint(C, x, a, b)[1], g)
        zs[i] = result[1]
        ns[i] = result[2]
        g = result[1]
    end
    return zs, ns    
end


function stat_test(n_samples::Int, n_perturb::Int, n_volume::Int, dx::Float64=1e-2)
    zs = zeros(ComplexF64, n_samples, n_perturb)
    ns = zeros(Int, n_samples, n_perturb)
    for i in 1:n_samples
		println("Sample " * string(i))
        while true
			result = zeros_perturbed(n_perturb, n_volume, dx)
            zs[i, :] = result[1]
            ns[i, :] = result[2]
            try
                result = zeros_perturbed(n_perturb, n_volume, dx)
                zs[i, :] = result[1]
                ns[i, :] = result[2]
                break
            catch e
				@show e
				println("Restarting at sample " * string(i))
                continue
            end
        end
    end
    return zs, ns    
end


n_samples = 10
n_perturb = 10
n_volume = 2
dx=1e-3
T = CUDA.rand(ComplexF64, 3 * n_volume^3) .- ComplexF64(0.5 + 0.5im)

domains = [[1, 2, 3], [5, 7, 8], [1, 4, 9, 10], [4, 5]]

p = OptProblem(n_volume, domains)
a = ComplexF64.(CUDA.Diagonal(CUDA.rand(Float64, size(p.G, 2))))
b = ComplexF64.(CUDA.Diagonal(CUDA.rand(Float64, size(p.G, 2))))
@show lagrangian(p, T, a, b)

#zs, ns = stat_test(n_samples, n_perturb, n_volume, dx)
#serialize("data/zeros_"*string(n_volume)*"_"*string(dx)*".dat", zs)
#serialize("data/iters_"*string(n_volume)*"_"*string(dx)*".dat", ns)

