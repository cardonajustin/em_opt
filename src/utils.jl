module utils
using CUDA, LinearAlgebra, GilaElectromagnetics, JLD2, BaryRational
export powm_gpu, bicgstab_gpu, load_greens_operator, pade_root


function bicgstab_gpu(A, b::CuArray{T}; tol=1e-7, max_iter=length(b)) where T
    n = length(b)
    x = CUDA.zeros(T, n)
    r = b - A(x)
    r_hat = copy(r)
    rho_old = 1.0
    alpha = 1.0
    omega_old = 1.0
    v = CUDA.zeros(T, n)
    p = CUDA.zeros(T, n)

    for iter in 1:max(max_iter, 1000)
        # println("\t"*string(norm(r)))
        rho_new = dot(r_hat, r)
        if iter == 1
            p = r
        else
            beta = (rho_new / rho_old) * (alpha / omega_old)
            p = r + beta * (p - omega_old * v)
        end

        v = A(p)
        alpha = rho_new / dot(r_hat, v)
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
    error("BiCGStab did not converge: $(norm(r))")
end
Base.:\(f::Function, x::CuArray{ComplexF64}) = bicgstab_gpu(f, x)


function powm_gpu(A, B, n, s = 0.0, tol::Float64=1e-1, max_iter=1000)
    x = CUDA.randn(ComplexF64, n)
    x ./= norm(x)
    λ_old = 0.0
    @show s
    M = x-> A(x) - s * B(x)
    for _ in 1:max_iter
        v = M(x)
        x,_ = B \ v
        x ./= norm(x)

        λ = dot(x, M(x)) / dot(x, B(x))
        λ_shifted = λ + s

        abs(λ_shifted - λ_old) > tol || return λ_shifted, x
        λ_old = λ_shifted
    end
    error("power method did not converge")
end


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


function pade_root(f::Function, z_init::Float64; max_iter::Int=5, max_restart::Int=5, r::Float64=1.0, tol=eps(Float32))
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
            p = maximum(real.(poles))
            z = maximum(real.(zeros))
            err, mvp = f(z)
            function_evaluations += mvp
            # println("\t\terr: "*string(err)*" "*string(mvps))
            domain = vcat(domain, [z])
            codomain = vcat(codomain, [err])
            abs(err) > tol || return z, p, iter, function_evaluations
        end
        # println("\tPPD did not converge, resampling")
		z_init = domain[argmin(abs.(codomain))]
    end
    error("Pade Zero Finder did not converge")
end
end