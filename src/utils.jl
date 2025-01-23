module utils
using CUDA, LinearAlgebra, GilaElectromagnetics, JLD2, BaryRational
export fun_to_mat, load_greens_operator, powm_gpu, pade_root


similar_fill(v::AbstractArray{T}, fill_val::T) where T = fill!(similar(v), fill_val)
similar_fill(v::AbstractArray{T}, dims::NTuple{N, Int}, fill_val::T) where {N, T} = fill!(similar(v, dims), fill_val)
Base.:\(::Nothing, x::AbstractArray) = (x, 0)
function lmul_mvp!(y::AbstractVector, ::Nothing, x::AbstractVector)
	copyto!(y, x)
	return 0
end

function powm_gpu(A, B, n, s = 0, tol=1e-1, max_iter=1000)
   x = CUDA.randn(ComplexF64, n)
   x ./= norm(x)
   λ_old = 0.0
   M = x-> A(x) - s * B(x)
   for _ in 1:max_iter
	   v = M(x)
	   x,_ = B \ v
	   x ./= norm(x)
	   λ = dot(x, M(x)) / dot(x, B(x))
	   λ_shifted = λ + s
	   if abs(λ_shifted - λ_old) < tol
		   return λ_shifted, x
	   end
	   λ_old = λ_shifted
   end
   error("Power method did not converge in $max_iter iterations.")
end

function bicgstab_gpu!(x::AbstractVector, op, b::AbstractVector; preconditioner=nothing, max_iter::Int=max(1000, size(op, 2)), atol::Real=zero(real(eltype(b))), rtol::Real=eps(real(eltype(b))), verbose::Bool=false, initial_zero::Bool=false)

    T = eltype(b)

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

    for num_iter in 1:max_iter
        # println("\t", norm(residual))
        norm(residual) > atol || return x, mvp

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
        residual -= ω*t
        ρ_prev = ρ
        !verbose || println(num_iter, " ", norm(residual))
    end
    error("BiCGStab did not converge after $max_iter iterations at $x.")
end

function bicgstab_gpu(op, b::AbstractVector; preconditioner=nothing, max_iter::Int=max(1000, length(b)), atol::Real=zero(real(eltype(b))), rtol::Real=eps(real(eltype(b))), verbose::Bool=false)
	x = similar_fill(b, zero(eltype(b)))
	return bicgstab_gpu!(x, op, b; preconditioner=preconditioner, max_iter=max_iter, atol=atol, rtol=rtol, verbose=verbose, initial_zero=true)
end
Base.:\(f::Function, x::AbstractArray) = bicgstab_gpu(f, x)


function load_greens_operator(cells::NTuple{3, Int}, scale::NTuple{3, Rational{Int}}; preload_dir="data")
    fname = "$(cells[1])x$(cells[2])x$(cells[3])_$(scale[1].num)ss$(scale[1].den)x$(scale[2].num)ss$(scale[2].den)x$(scale[3].num)ss$(scale[3].den).jld2"
    fpath = joinpath(preload_dir, fname)
    if isfile(fpath)
            file = jldopen(fpath)
            fourier = CuArray.(file["fourier"])
            options = GlaKerOpt(true)
            volume = GlaVol(cells, scale, (0//1, 0//1, 0//1))
            mem = GlaOprMem(options, volume; egoFur=fourier, setTyp=ComplexF64)
            return GlaOpr(mem)
    end
    operator = GlaOpr(cells, scale; setTyp=ComplexF64, useGpu=true)
    fourier = Array.(operator.mem.egoFur)
    jldsave(fpath; fourier=fourier)
    return operator
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
            a = aaa(domain, codomain, clean=1)
            poles, _, zeros = prz(a)
            z = maximum(real.(zeros))
            p = maximum(real.(poles))
            err, mvp = f(z)
            function_evaluations += mvp
            abs(err) > tol || return z, p, iter, function_evaluations
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