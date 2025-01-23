module opt_problem
include("utils.jl")
using .utils, GilaElectromagnetics, CUDA, LinearAlgebra
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
	return pade_root(x-> partial_dual_constraint(prob, x, a, b)[1:3], g + sqrt(abs(g)), max_iter=10)
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






similar_fill(v::AbstractArray{T}, fill_val::T) where T = fill!(similar(v), fill_val)
similar_fill(v::AbstractArray{T}, dims::NTuple{N, Int}, fill_val::T) where {N, T} = fill!(similar(v, dims), fill_val)
Base.:\(::Nothing, x::AbstractArray) = (x, 0)
function lmul_mvp!(y::AbstractVector, ::Nothing, x::AbstractVector)
	copyto!(y, x)
	return 0
end

function partial_dual!(x::AbstractVector, prob::OptProblem; preconditioner=nothing, max_iter::Int=max(1000, length(prob.d) * 2), atol::Real=zero(Float64), rtol::Real=eps(Float64), verbose::Bool=false, initial_zero::Bool=false)

	n = size(prob.d, 1) * 2
    T = Float64
    mvp = 0

	g, pole, _, _ = partial_dual_root(prob, x[1:size(prob.d, 1)], x[size(prob.d, 1)+1:end])
	op = a-> vcat(partial_dual_fn_dl(prob, a[1:size(prob.d, 1)], a[size(prob.d, 1)+1:end], g)...)


	b = zeros(Float64, n)
    ρ_prev = zero(T)
    ω = zero(T)
    α = zero(T)
    v = zeros(Float64, n)
    residual = deepcopy(b)
    if !initial_zero
	    residual = b - op(x)
	    mvp += 1
    end
    atol = max(atol, rtol * norm(residual))
    residual_shadow = deepcopy(residual)
    p = deepcopy(residual)
    s = similar(residual)

	x_prev = deepcopy(x)

    for num_iter in 1:max_iter
		println(norm(residual), " ", g - pole)
		if g - pole < 1e-0
			println("Restarting with fake sources.")
			LTT, _, _ = partial_dual_matrices(prob, 0.0, x[1:size(prob.d, 1)], x[size(prob.d, 1)+1:end])
			_, v = powm_gpu(LTT, x-> I * x, size(prob.G, 2))
			prob.S += v
			return partial_dual!(x_prev, prob; preconditioner=preconditioner, max_iter=max_iter, atol=atol, rtol=rtol, verbose=verbose, initial_zero=false)
		end
        norm(residual) > atol || return x, mvp
		g, pole, _, _ = pade_root(z -> partial_dual_constraint(prob, z, x[1:size(prob.d, 1)], x[size(prob.d, 1)+1:end])[1:3], g)
		op = a-> vcat(partial_dual_fn_dl(prob, a[1:size(prob.d, 1)], a[size(prob.d, 1)+1:end], g)...)

        ρ = dot(residual_shadow, residual)
        if num_iter > 1
            β = (ρ / ρ_prev) * (α / ω)
            p = residual + β*(p - ω*v)
        end
        p̂, precon_mvp = preconditioner \ p
        mvp += precon_mvp
        v = op(p̂)
        mvp += 1
        residual_v = dot(residual_shadow, v)
        α = ρ / residual_v
        residual -= α*v
        s = deepcopy(residual)

        norm(residual) > atol || return x + α*p̂, mvp

        ŝ, precon_mvp = preconditioner \ s
        mvp += precon_mvp
        t = op(ŝ)
        mvp += 1
        ω = dot(t, s) / dot(t, t)
        x += α*p̂ + ω*ŝ
		x_prev = deepcopy(x)
        residual -= ω*t
        ρ_prev = ρ
        !verbose || println(num_iter, " ", norm(residual))
    end
    error("Partial Dual did not converge after $max_iter iterations at $x.")
end

function partial_dual(prob::OptProblem; preconditioner=nothing, max_iter::Int=max(1000, length(prob.d) * 2), atol::Real=zero(Float64), rtol::Real=eps(Float64), verbose::Bool=false)
	# x = similar_fill(b, zero(eltype(b)))
	n = size(prob.d, 1) * 2
	x = randn(Float64, n)
	return partial_dual!(x, prob; preconditioner=preconditioner, max_iter=max_iter, atol=atol, rtol=rtol, verbose=verbose, initial_zero=false)
end
end