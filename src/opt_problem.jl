module opt_problem
include("utils.jl")
using .utils, CUDA, LinearAlgebra, GilaElectromagnetics
export OptProblem, objective, constraint, partial_dual_constraint, partial_dual_root, partial_dual_fn_dl, partial_dual_fn, partial_dual


sym = M::AbstractMatrix -> 0.5 * (M + adjoint(M))
asym = M::AbstractMatrix -> -0.5im * (M - adjoint(M))


mutable struct OptProblem
	G::GlaOpr
	X::Diagonal{ComplexF64, CuArray{ComplexF64, 1, CUDA.DeviceMemory}}
	Q::Function
	S::CuArray{ComplexF64}
	d::Vector{Vector{Int}}
end


function objective(prob::OptProblem, T::CuArray{ComplexF64})
	return real(imag(prob.S' * T) - T' * prob.Q(T))
end


function constraint(prob::OptProblem, i::Int64, T::CuArray{ComplexF64})
    P = CUDA.zeros(size(prob.G, 2))
	P[prob.d[i]] .= 1.0
	PT = P .* T
    return prob.S' * PT +  T' * (adjoint(inv(prob.X)) * PT - adjoint(prob.G) * PT)
end


function multiplier_matrix(prob::OptProblem, a::Array{Float64})
	A = zeros(ComplexF64, size(prob.X, 1))
	B = zeros(ComplexF64, size(prob.X, 1))
	for i in 1:size(prob.d, 1)
		A[prob.d[i]] .+= ComplexF64(a[i])
		B[prob.d[i]] .+= ComplexF64(a[i + size(prob.d, 1)])
	end
	return CUDA.Diagonal(CuArray(A)), CUDA.Diagonal(CuArray(B))
end


function partial_dual_matrices(prob::OptProblem, z::Float64, a::Array{Float64})
	A, B = multiplier_matrix(prob, a)
    l = A - im*(B + z*I)
    LTT = x-> prob.Q(x) + sym(adjoint(inv(prob.X)) * l) * x - 0.5 * (adjoint(prob.G) * (l * x)) - 0.5 * (adjoint(l) * (prob.G * x))
    LTS = 0.5 * conj(l)
    E = x-> asym(adjoint(inv(prob.X))) * x + 0.5im * (adjoint(prob.G) * x - prob.G * x)
    return LTT, LTS, E
end


function partial_dual_constraint(prob::OptProblem, a::AbstractArray{Float64}, z::Float64)
    LTT, LTS, E = partial_dual_matrices(prob, z, a)
    T, mvp = LTT \ (LTS * prob.S)
    return real(imag(prob.S' * T) - T' * E(T)), mvp, T
end


function partial_dual_root(prob::OptProblem, a::AbstractArray{Float64})
    LTT, _, E = partial_dual_matrices(prob, 0.0, a)
	g, _ = powm_gpu(LTT, E, size(prob.G, 2))
	if real(g) > 0
		g, _ = powm_gpu(LTT, E, size(prob.G, 2), g)
	end
	g = -real(g)
	root, pole, _, mvp = pade_root(x-> partial_dual_constraint(prob, a, x)[1:3], g + sqrt(abs(g)), max_iter=10)
	return root, pole, mvp
end


function partial_dual_fn_dl(prob::OptProblem, a::AbstractArray{Float64}, g::Float64)
	_, _, T = partial_dual_constraint(prob, a, g)
	d = map(i-> constraint(prob, i, T), 1:size(prob.d, 1))
	return real(d), imag(d)
end


function partial_dual_fn(prob::OptProblem, a::AbstractArray{Float64}, g::Float64)
	c_re, c_im = partial_dual_fn_dl(prob, a, g)
	C, _, T = partial_dual_constraint(prob, a, g)
	return real(objective(prob, T) + sum(a[1:size(prob.d, 1)] .* c_re) + sum(a[size(prob.d, 1)+1:end] .* c_im)) + g * C
end


function partial_dual(prob::OptProblem; tol=1e-7, max_iter=size(prob.d, 1))
    T = Float64
    n = size(prob.d, 1) * 2
	g, p, _ = partial_dual_root(prob, ones(Float64, n))
	A = a-> vcat(partial_dual_fn_dl(prob, a, g)...)
    x = rand(T, n)
    b = zeros(T, n)
    r = b - A(x)
    r_hat = copy(r)
    rho_old = 1.0
    alpha = 1.0
    omega_old = 1.0
    v = zeros(T, n)
    p = zeros(T, n)

    try
        for iter in 1:max(max_iter, 1000)
            g, pole = pade_root(z-> partial_dual_constraint(prob, x, z)[1:3], g)
            g > pole || error("Exceeded domain of duality")
            A = a-> vcat(partial_dual_fn_dl(prob, a, g)...)
            rho_new = dot(r_hat, r)
            if iter == 1
                p = r
            else
                beta = (rho_new / rho_old) * (alpha / omega_old)
                p_prev = p
                p = r + beta * (p - omega_old * v)
            end

            v = A(p)
            alpha = rho_new / dot(r_hat, v)
            s_prev = s
            s = r - alpha * v
            real(norm(s)) > tol || return x + alpha * p, iter

            t = A(s)
            omega_new = dot(t, s) / dot(t, t)
            x += alpha * p + omega_new * s
            r = s - omega_new * t
            real(norm(r)) > tol || return x, iter

            rho_old = rho_new
            omega_old = omega_new
        end
    catch
        println("Exceeded domain of duality")
        LTT, _, _ = partial_dual_matrices(prob, g, x)
        s, _ = powm_gpu(LTT, x->I(x), size(prob.G, 2))
        _, v = powm_gpu(LTT, x->I(x), size(prob.G, 2), -s)
        prob.S += v
        return partial_dual(prob, tol=tol, max_iter=max_iter)
    end
    error("Dual did not converge: $(norm(r))")
end


end