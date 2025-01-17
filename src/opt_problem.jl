module opt_problem
include("utils.jl")
using .utils, GilaElectromagnetics, CUDA, LinearAlgebra, Roots, JacobiDavidson, BaryRational
export OptProblem, objective, constraint_re, constraint_im, lagrangian, get_LTT_E, partial_dual_constraint, partial_dual_root, partial_dual_fn, partial_dual_fn_dl, partial_dual, pade_root

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
    ASym = M::AbstractMatrix -> -0.5im * (M - adjoint(M))
	P = CUDA.zeros(size(prob.G, 2))
	P[prob.d[i]] .= 1.0
	P = CUDA.Diagonal(P)
	return real(imag(prob.S' * (P * T)) +  T' * (ASym(adjoint(inv(prob.X))* P) * T) + 0.5*im * (T' * (adjoint(prob.G)*T) - T' * (P * (prob.G*T))))
end


function constraint_im(prob::OptProblem, i::Int64, T::AbstractArray{ComplexF64})
    Sym = M::AbstractMatrix -> 0.5 * (M + adjoint(M))
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


function get_LTT_E(prob::OptProblem, z::Float64, A::AbstractArray{Float64}, B::AbstractArray{Float64})
	a, b = multiplier_matrix(prob, A, B)
    Sym = M::AbstractMatrix -> 0.5 * (M + adjoint(M))
    ASym = M::AbstractMatrix -> -0.5im * (M - adjoint(M))
    l = a - im*(b + z*I)
    LTT = x-> prob.Q*x + Sym(adjoint(inv(prob.X))*l)*x - 0.5 * (adjoint(prob.G)*(l*x)) - 0.5 * (adjoint(l)*(prob.G*x))
    E = x-> ASym(adjoint(inv(prob.X)))*x + 0.5*im * (adjoint(prob.G)*x - prob.G*x)
    LTS = 0.5 * (a + im * (b + z * I))
    return LTT, E, LTS
end


function partial_dual_constraint(prob::OptProblem, z::Float64, a::AbstractArray{Float64}, b::AbstractArray{Float64})
    LTT, E, LTS = get_LTT_E(prob, z, a, b)
    T, mvp = LTT \ (LTS * prob.S)
    return real(imag(prob.S' * T) - T' * E(T)), mvp, T
end


function partial_dual_root(prob::OptProblem, a::AbstractArray{Float64}, b::AbstractArray{Float64})
    LTT, E, _ = get_LTT_E(prob, 0.0, a, b)
	g, _ = powm_gpu(LTT, E, size(prob.G, 2))
	if real(g) > 0
		g, _ = powm_gpu(LTT, E, size(prob.G, 2), g)
	end
	g = -real(g)
	return pade_root(x-> partial_dual_constraint(prob, x, a, b)[1], g + sqrt(abs(g)), max_iter=10)
end


function partial_dual_fn(prob::OptProblem, a::AbstractArray{Float64}, b::AbstractArray{Float64}, g::Float64)
	C, _, T = partial_dual_constraint(prob, g, a, b)
	return lagrangian(prob, T, a, b) + g * C
end

function partial_dual_fn_dl(prob::OptProblem, a::AbstractArray{Float64}, b::AbstractArray{Float64}, g::Float64)
	_, _, T = partial_dual_constraint(prob, g, a, b)
	d_re = map(i-> constraint_im(prob, i, T), 1:size(a, 1))
	d_im = map(i-> constraint_im(prob, i, T), 1:size(b, 1))
	#d_re = zeros(Float64, size(a, 1))
	#d_im = zeros(Float64, size(b, 1))
	#for i in 1:size(prob.d, 1)
	#	d_re[i] = constraint_re(prob, i, T)
	#	d_im[i] = constraint_im(prob, i, T)
	#end
	return d_re, d_im
end


function partial_dual(prob::OptProblem, a0::AbstractArray{Float64}, b0::AbstractArray{Float64}, g0::Float64, tol::Float64=1e-10, max_iter::Int=5, learning_rate::Float64=1e-0)
	a = a0
	b = b0
	g = g0
	for iter in 1:max_iter
		g_temp = pade_root(z-> partial_dual_constraint(prob, z, a, b)[1], g)[1]
		gradient = partial_dual_fn_dl(prob, a, b, g_temp)
		err = norm(gradient)
		err > tol / size(gradient, 1) || break
		@show err
		a -= (learning_rate) * gradient[1]
		b -= (learning_rate) * gradient[2]
		g = g_temp
	end
	return partial_dual_fn(prob, a, b, g)
end


function pade_root(f, z_init::Float64; n_init::Int=1, max_iter::Int=5, max_restart::Int=5, r::Float64=1e-2, tol=eps(Float32))
    inverse_solves = 0
    err = 0
    for _ in 0:max_restart
		r = min(abs(err), r)
		err_init = f(z_init)
		z = rand(Float64) + z_init - 0.5
		if err > 0
			z -= 1.0
		end
		err = f(z)
        domain = [z, z_init]
        codomain = [err, err_init]
        inverse_solves += n_init
        for _ in 1:max_iter
            abs(err) > tol || return z, inverse_solves
            a = aaa(domain, codomain, clean=1)
            _, _, zeros = prz(a)
            z = maximum(real.(zeros))
            err = f(z)
            inverse_solves += 1
            println("\t\terr: "*string(err))
            domain = vcat(domain, [z])
            codomain = vcat(codomain, [err])
        end
        println("\tPPD did not converge, resampling")
		z_init = domain[argmin(abs.(codomain))]
    end
    throw("Pade Zero Finder did not converge")
end
end
