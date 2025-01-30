module utils
    using GilaElectromagnetics, CUDA, LinearAlgebra, JLD2, BaryRational
    export sym, asym, load_greens_operator, powm_gpu, pade_root

    sym  = M::AbstractMatrix ->  0.5   * (M + M')
    asym = M::AbstractMatrix -> -0.5im * (M - M')


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


    function bicgstab_gpu(A::Function, b::AbstractArray{ComplexF64}; x0::AbstractArray{ComplexF64}=CUDA.randn(ComplexF64, length(b)), max_iter::Int=1000, tol::Float64=1e-9)
        n = length(b)
        x = copy(x0)
        r = b - A(x)
        r0 = copy(r)
        rho = 1.0
        alpha = 1.0
        omega = 1.0
        v = CUDA.zeros(ComplexF64, n)
        p = CUDA.zeros(ComplexF64, n)
        
        for iter in 1:max_iter
            if norm(r) < tol
                return x
            end
            rho_prev = rho
            rho = dot(r0, r)
            beta = (rho / rho_prev) * (alpha / omega)
            p = r + beta * (p - omega * v)
            v = A(p)
            alpha = rho / dot(r0, v)
            s = r - alpha * v
            t = A(s)
            omega = dot(t, s) / dot(t, t)
            x = x + alpha * p + omega * s
            r = s - omega * t
        end
        error("BiCGStab did not converge within $max_iter iterations")
     
    end
    Base.:\(f::Function, x::AbstractArray) = bicgstab_gpu(f, x)


    function powm_gpu(A, B, n, s = 0.0, tol=1e-1, max_iter=1000)
        x = CUDA.randn(ComplexF64, n)
        x ./= norm(x)
        λ_old = 0.0
        M = x-> A(x) - s * B(x)
        for _ in 1:max_iter
            v = M(x)
            x = B \ v
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


    function pade_root(f, z_init::Float64; max_iter::Int=5, max_restart::Int=5, r::Float64=1.0, tol=eps(Float32))
        err = 0
        for _ in 0:max_restart
            err_init = f(z_init)
            r = min(abs(err_init), r)
            z = rand(Float64) + z_init + sign(err_init) * r
            err = f(z)
            domain = [z, z_init]
            codomain = [err, err_init]
            for iter in 1:max_iter
                a = aaa(domain, codomain, clean=1)
                poles, _, zeros = prz(a)
                z = maximum(real.(zeros))
                p = maximum(real.(poles))
                err = f(z)
                abs(err) > tol || return z, p
                println("\terr: "*string(err))
                domain = vcat(domain, [z])
                codomain = vcat(codomain, [err])
            end
            println("Resampling")
            z_init = domain[argmin(abs.(codomain))]
        end
        error("Pade Zero Finder did not converge")
    end
end