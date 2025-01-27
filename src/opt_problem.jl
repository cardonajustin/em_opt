module opt_problem
include("utils.jl")
using .utils, GilaElectromagnetics, CUDA, LinearAlgebra
export OptProblem, dual


mutable struct OptProblem
	G::GlaOpr
	X::AbstractMatrix{ComplexF64}
	Oquad::AbstractMatrix{ComplexF64}
	Olin::AbstractArray{ComplexF64}
	Ei::AbstractArray{ComplexF64}
	d::Vector{Vector{Int}}
end


function symU(prob::OptProblem, T::AbstractArray{ComplexF64}; P::AbstractMatrix{ComplexF64}=CUDA.Diagonal(CUDA.ones(ComplexF64, size(prob.G, 2))))
	return sym(inv(prob.X)' * P) * T - 0.5 * (prob.G' * (P * T) + P * (prob.G * T))
end


function asymU(prob::OptProblem, T::AbstractArray{ComplexF64}; P::AbstractMatrix{ComplexF64}=CUDA.Diagonal(CUDA.ones(ComplexF64, size(prob.G, 2))))
	return asym(inv(prob.X)' * P) * T + 0.5im * (prob.G' * (P * T) - P * (prob.G * T))
end


function Lquad(prob::OptProblem, T::AbstractArray{ComplexF64}, l::AbstractArray{Float64}, g::Float64)
	result = prob.Oquad * T + g * asymU(prob, T)
	for i in 1:length(prob.d)
		P = zeros(ComplexF64, size(prob.X))
		P[prob.d[i]] .= 1.0
		P = CUDA.Diagonal(CuArray(P))
		result += l[i] * symU(prob, T, P=P)
		result += l[i + length(prob.d)] * asymU(prob, T, P=P)
	end
	return result
end


function Llin(prob::OptProblem, l::AbstractArray{Float64}, g::Float64)
	coef = 1im * g 
	coef += sum(l[1:length(prob.d)])
	coef += 1im * sum(l[length(prob.d)+1:end])
	return prob.Olin + coef * prob.Ei
end


function partial_constraint(prob::OptProblem, l::AbstractArray{Float64}, g::Float64)
	L = T-> Lquad(prob, T, l, g)
	S = Llin(prob, l, g)
	T = L \ S
	return imag(prob.Ei' * T) + real(T' * asymU(prob, T))
end


function partial_multiplier(prob::OptProblem, l::AbstractArray{Float64})
	L = T-> Lquad(prob, T, l, 0.0)
	E = T-> asymU(prob, T)
	g = powm_gpu(L, E, size(prob.G, 2))
	if real(g) > 0
		g = powm_gpu(L, E, size(prob.G, 2), g)
	end
	g = -real(g)
	return pade_root(x-> partial_constraint(prob, l, x), g + sqrt(abs(g)), max_iter=10)
	
end


function dual_grad(prob::OptProblem, l::AbstractArray{Float64}, g::Float64)
	L = T-> Lquad(prob, T, l, g)
	S = Llin(prob, l, g)
	T = L \ S
	result = zeros(Float64, length(prob.d) * 2)
	for i in 1:length(prob.d)
		P = zeros(ComplexF64, size(prob.X))
		P[prob.d[i]] .= 1.0
		P = CUDA.Diagonal(CuArray(P))
		result[i] = real(prob.Ei' * T) - real(T' * symU(prob, T, P=P))
		result[i + length(prob.d)] = imag(prob.Ei' * T) - real(T' * asymU(prob, T, P=P))
	end
	return result
end


similar_fill(v::AbstractArray{T}, fill_val::T) where T = fill!(similar(v), fill_val)
similar_fill(v::AbstractArray{T}, dims::NTuple{N, Int}, fill_val::T) where {N, T} = fill!(similar(v, dims), fill_val)
Base.:\(::Nothing, x::AbstractArray) = (x, 0)
function lmul_mvp!(y::AbstractVector, ::Nothing, x::AbstractVector)
	copyto!(y, x)
	return 0
end

function dual(prob::OptProblem, x = randn(Float64, length(prob.d) * 2); preconditioner=nothing, max_iter::Int=max(1000, size(prob.G, 2)), atol::Real=zero(Float64), rtol::Real=eps(Float64), verbose::Bool=false, initial_zero::Bool=false)

    T = Float64
	b = zeros(T, length(prob.d) * 2)
	# g, pole = partial_multiplier(prob, x)
	op = l-> dual_grad(prob, l, 0.0)

    mvp = 0

    ρ_prev = zero(T)
    ω = zero(T)
    α = zero(T)
    v = similar_fill(b, zero(T))
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
		g, pole = partial_multiplier(prob, x)
        println("\t", norm(residual), " ", g - pole)
		if g - pole < 1.0
			println("Restarting with fake sources")
			L = T-> Lquad(prob, T, x, g)
			S = Llin(prob, x, g)
			E_fake = L \ S
			prob.Ei += E_fake
			return dual(prob, x_prev)
		end
        norm(residual) > atol || return x
		# g, pole = pade_root(z-> partial_constraint(prob, x, z), g)
		# op = l-> dual_grad(prob, l, g)

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

        norm(residual) > atol || return x + α*p̂

        ŝ, precon_mvp = preconditioner \ s
        mvp += precon_mvp
        t = op(ŝ)
        mvp += 1
        ω = dot(t, s) / dot(t, t)
		x_prev = deepcopy(x)
        x += α*p̂ + ω*ŝ
        residual -= ω*t
        ρ_prev = ρ
        !verbose || println(num_iter, " ", norm(residual))
    end
    error("BiCGStab did not converge after $max_iter iterations at $x.")
end

end