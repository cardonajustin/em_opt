module opt_problem
include("utils.jl")
using .utils, GilaElectromagnetics, CUDA, LinearAlgebra, JacobiDavidson, BaryRational
export OptProblem, objective, constraint_re, constraint_im, lagrangian, partial_dual_matrices, partial_dual_constraint, partial_dual_root, partial_dual_fn, partial_dual_fn_dl, partial_dual, pade_root


Sym = M::AbstractMatrix -> 0.5 * (M + adjoint(M))
ASym = M::AbstractMatrix -> -0.5im * (M - adjoint(M))


mutable struct OptProblem
	G::GlaOpr
	X::AbstractMatrix{ComplexF64}
	Q::AbstractMatrix{ComplexF64}
	S::AbstractArray{ComplexF64}
	d::Vector{Vector{Int}}
end


function OptProblem(n::Int64, d::Vector{Vector{Int}})
    G = load_greens_operator((n, n, n), (1//32, 1//32, 1//32))
    m = size(G, 2)

    X = CUDA.Diagonal(ComplexF64(rand(Float64) - 0.5 + 1e-3im * rand(Float64)) .* CUDA.ones(ComplexF64, m))
    Q = CUDA.Diagonal(ComplexF64.(CUDA.ones(Float64, m)))    
    S = CUDA.rand(ComplexF64, m) .- ComplexF64(0.5 + 0.5im)
    return OptProblem(G, X, Q, S, d)
end


function objective(prob::OptProblem, T::AbstractArray{ComplexF64})
	return real(imag(prob.S' * T) - T' * (prob.Q * T))
end


function constraint_re(prob::OptProblem, i::Int64, T::AbstractArray{ComplexF64})
	P = CUDA.zeros(size(prob.G, 2))
	P[prob.d[i]] .= 1.0
	P = CUDA.Diagonal(P)
	return real(imag(prob.S' * (P * T)) +  T' * (ASym(adjoint(inv(prob.X))* P) * T) + 0.5*im * (T' * (adjoint(prob.G)*T) - T' * (P * (prob.G*T))))
end


function constraint_im(prob::OptProblem, i::Int64, T::AbstractArray{ComplexF64})
	P = CUDA.zeros(size(prob.G, 2))
	P[prob.d[i]] .= 1.0
	P = CUDA.Diagonal(P)
	return real(real(prob.S' * (P * T)) +  T' * (Sym(adjoint(inv(prob.X)) * P) * T) - 0.5 * (T' * (adjoint(prob.G)*(P * T)) + T' * (P * (prob.G*T))))
end


function multiplier_matrix(prob::OptProblem, a::AbstractArray{Float64}, b::AbstractArray{Float64})
	A = zeros(ComplexF64, size(prob.X, 1))
	B = zeros(ComplexF64, size(prob.X, 1))
	for i in 1:size(prob.d, 1)
		A[prob.d[i]] .+= ComplexF64(a[i])
		B[prob.d[i]] .+= ComplexF64(b[i])
	end
	return CUDA.Diagonal(CuArray(A)), CUDA.Diagonal(CuArray(B))
end


function lagrangian(prob::OptProblem, T::AbstractArray{ComplexF64}, a::AbstractArray{Float64}, b::AbstractArray{Float64})
	c_re, c_im = partial_dual_fn_dl(prob, a, b, 0.0)
	return real(objective(prob, T) + sum(a .* c_re) + sum(b .* c_im))
end


function partial_dual_matrices(prob::OptProblem, z::Float64, A::AbstractArray{Float64}, B::AbstractArray{Float64})
	a, b = multiplier_matrix(prob, A, B)
    l = a - im*(b + z*I)
    LTT = x-> prob.Q*x + Sym(adjoint(inv(prob.X))*l)*x - 0.5 * (adjoint(prob.G)*(l*x) + adjoint(l)*(prob.G*x))
    LTS = 0.5 * (a + im * (b + z * I))
    E = x-> ASym(adjoint(inv(prob.X)))*x + 0.5*im * (adjoint(prob.G)*x - prob.G*x)
    return LTT, LTS, E
end


function partial_dual_constraint(prob::OptProblem, z::Float64, a::AbstractArray{Float64}, b::AbstractArray{Float64})
    LTT, LTS, E = partial_dual_matrices(prob, z, a, b)
    T, mvp = LTT \ (LTS * prob.S)
    return real(imag(prob.S' * T) - T' * E(T)), mvp, T
end


function partial_dual_root(prob::OptProblem, a::AbstractArray{Float64}, b::AbstractArray{Float64})
    LTT, _, E = partial_dual_matrices(prob, 0.0, a, b)
	g, _ = powm_gpu(LTT, E, size(prob.G, 2))
	if real(g) > 0
		g, _ = powm_gpu(LTT, E, size(prob.G, 2), g)
	end
	g = -real(g)
	root, mvp = pade_root(x-> partial_dual_constraint(prob, x, a, b)[1:3], g + sqrt(abs(g)), max_iter=10)
	return root, mvp
end


function partial_dual_fn(prob::OptProblem, a::AbstractArray{Float64}, b::AbstractArray{Float64}, g::Float64)
	C, _, T = partial_dual_constraint(prob, g, a, b)
	return lagrangian(prob, T, a, b) + g * C
end


function partial_dual_fn_dl(prob::OptProblem, a::AbstractArray{Float64}, b::AbstractArray{Float64}, g::Float64)
	_, _, T = partial_dual_constraint(prob, g, a, b)
	d_re = map(i-> constraint_im(prob, i, T), 1:size(a, 1))
	d_im = map(i-> constraint_im(prob, i, T), 1:size(b, 1))
	return d_re, d_im
end


function partial_dual(prob::OptProblem, a0::AbstractArray{Float64}, b0::AbstractArray{Float64}, g0::Float64, tol::Float64=1e-10, max_iter::Int=5, learning_rate::Float64=1e-0)
	a = a0
	b = b0
	g = g0
	for _ in 1:max_iter
		g_temp, _, _ = pade_root(z-> partial_dual_constraint(prob, z, a, b)[1:3], g)
		grad_a, grad_b = partial_dual_fn_dl(prob, a, b, g_temp)
		err = norm(grad_a) + norm(grad_b)
		err > tol / (size(grad_a, 1) + size(grad_b, 1)) || break
		a -= (learning_rate) * grad_a
		b -= (learning_rate) * grad_b
		g = g_temp
	end
	return partial_dual_fn(prob, a, b, g)
end


function pade_root(f, z_init::Float64; max_iter::Int=5, max_restart::Int=5, r::Float64=1.0, tol=eps(Float32))
    function_evaluations = 0
    err = 0
    for _ in 0:max_restart
		err_init, mvp = f(z_init)
		function_evaluations += mvp
		r = min(abs(err_init), r)
		z = rand(Float64) + z_init + sign(err_init) * r
		err, mvp = f(z)
		function_evaluations += mvp
        domain = [z, z_init]
        codomain = [err, err_init]        
        for iter in 1:max_iter
            abs(err) > tol || return z, iter, function_evaluations
            a = aaa(domain, codomain, clean=1)
            _, _, zeros = prz(a)
            z = maximum(real.(zeros))
            err, mvp = f(z)
            function_evaluations += mvp
            # println("\t\terr: "*string(err)*" "*string(mvps))
            domain = vcat(domain, [z])
            codomain = vcat(codomain, [err])
        end
        # println("\tPPD did not converge, resampling")
		z_init = domain[argmin(abs.(codomain))]
    end
    error("Pade Zero Finder did not converge")
end
end