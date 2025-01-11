module constraint_function
include("utils.jl")
using .utils, GilaElectromagnetics, CUDA, LinearAlgebra, Roots, JacobiDavidson, BaryRational
export ConstraintFunction, get_LTT_E, partial_dual_root, pade_root

mutable struct ConstraintFunction
	G::GlaOpr
	X::AbstractMatrix{ComplexF64}
	Q::AbstractMatrix{ComplexF64}
	S::AbstractArray{ComplexF64}
	a::AbstractMatrix{ComplexF64}
	b::AbstractMatrix{ComplexF64}
end


function ConstraintFunction(n::Int=128)
    G = load_greens_operator((n, n, n), (1//32, 1//32, 1//32))
    m = size(G, 2)

    X = CUDA.Diagonal(ComplexF64(rand(Float64) - 0.5 + 1e-3im * rand(Float64)) .* CUDA.ones(ComplexF64, m))
    Q = CUDA.Diagonal(ComplexF64.(CUDA.ones(Float64, m)))    
    S = CUDA.rand(ComplexF64, m) .- ComplexF64(0.5 + 0.5im)
    a = CUDA.Diagonal(ComplexF64.(CUDA.ones(Float64, m)))
    b = CUDA.Diagonal(ComplexF64.(CUDA.zeros(Float64, m)))
    return ConstraintFunction(G, X, Q, S, a, b)
end


function get_LTT_E(C::ConstraintFunction, z::Float64)
    Sym = M::AbstractMatrix -> 0.5 * (M + adjoint(M))
    ASym = M::AbstractMatrix -> -0.5im * (M - adjoint(M))
    l = C.a - im*(C.b + z*I)
    LTT = x-> C.Q*x + Sym(adjoint(inv(C.X))*l)*x - 0.5 * (adjoint(C.G)*(l*x)) - 0.5 * (adjoint(l)*(C.G*x))
    E = x-> ASym(adjoint(inv(C.X)))*x + 0.5*im * (adjoint(C.G)*x - C.G*x)
    return LTT, E
end


function (C::ConstraintFunction)(z::Float64)
    LTT, E = get_LTT_E(C, z)
    LTS = 0.5 * (C.a + im * (C.b + z * I))
    T, mvp = LTT \ (LTS * C.S)
    return real(imag(C.S' * T) - T' * E(T)), mvp
end


function partial_dual_root(C::ConstraintFunction)
    LTT, E = get_LTT_E(C, 0.0)
	g, _ = powm_gpu(LTT, E, size(C.G, 2))
	if real(g) > 0
		g, _ = powm_gpu(LTT, E, size(C.G, 2), g)
	end
	g = -real(g)
	return pade_root(x-> C(x)[1], g + sqrt(abs(g)))
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
