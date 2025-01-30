module opt_problem
    include("../src/utils.jl")
    using .utils, GilaElectromagnetics, CUDA, LinearAlgebra
    export OptProblem, dual


    mutable struct OptProblem
        Ol::AbstractArray{ComplexF64}
        Oq::AbstractMatrix{ComplexF64}
        d::Vector{Vector{Int}}
        Ei::AbstractArray{ComplexF64}
        sU::Function
        aU::Function
        FS::AbstractArray{ComplexF64}
    end


    function OptProblem(Ol::AbstractArray{ComplexF64}, Oq::AbstractMatrix{ComplexF64}, d::Vector{Vector{Int}}, Ei::AbstractArray{ComplexF64}, G::GlaOpr, X::AbstractMatrix{ComplexF64})
        m = length(Ol)
        id = CUDA.Diagonal(CUDA.ones(ComplexF64, m))
        sU = (P::AbstractArray{ComplexF64}, proj::AbstractMatrix{ComplexF64}=id) ->  sym(inv(X)' * proj) * P - 0.5   * (G' * (proj * P) + proj * (G * P))
        aU = (P::AbstractArray{ComplexF64}, proj::AbstractMatrix{ComplexF64}=id) -> asym(inv(X)' * proj) * P + 0.5im * (G' * (proj * P) - proj * (G * P))
        return OptProblem(Ol, Oq, d, Ei, sU, aU, CUDA.zeros(ComplexF64, size(G, 2)))
    end


    function Ll(prob::OptProblem, l::AbstractArray{Float64}, g::Float64)
        n = length(prob.d)
        coef = sum(l[1:n]) + (sum(l[n+1:2*n]) + g) * 1im
        coef_fs = sum(l[2*n+1:3*n]) + sum(l[3*n+1:end]) * 1im
        return prob.Ol + coef * prob.Ei + coef_fs * prob.FS
    end


    function Lq(prob::OptProblem, P::AbstractArray{ComplexF64}, l::AbstractArray{Float64}, g::Float64)
        result = prob.Oq * P + g * prob.aU(P)
        for i in 1:length(prob.d)
            proj = zeros(ComplexF64, size(prob.Ol))
            proj[prob.d[i]] .= 1.0
            proj = CUDA.Diagonal(CuArray(proj))
            result += prob.sU(P, proj) * l[i]
            result += prob.aU(P, proj) * l[i + length(prob.d)]
            result += prob.sU(P, proj) * l[i + 2 * length(prob.d)]
            result += prob.aU(P, proj) * l[i + 3 * length(prob.d)]
        end
        return result
    end


    function partial_constraint(prob::OptProblem, l::AbstractArray{Float64}, g::Float64)
        Q = p-> Lq(prob, p, l, g)
        L = Ll(prob, l, g)
        P = 0.5 .* (Q \ L)
        return imag(prob.Ei' * P) + real(P' * prob.aU(P))
    end


    function partial_multiplier(prob::OptProblem, l::AbstractArray{Float64})
        Q = P-> Lq(prob, P, l, 0.0)
        g, _ = powm_gpu(Q, prob.aU, length(prob.Ol))
        if real(g) > 0
            g, _ = powm_gpu(Q, prob.aU, length(prob.Ol), g)
        end
        g = -real(g)
        if g < 0
            return 0.0, -Inf64
        end
        return pade_root(x-> partial_constraint(prob, l, x), g + log(g))
    end


    function eig_gpu(A, n, tol=1e-1, max_iter=1000)
        x = CUDA.randn(ComplexF64, n)
        x ./= norm(x)
        λ_old = 0.0
        for _ in 1:max_iter
            x = A \ x
            v = copy(x)
            x ./= norm(x)
            λ = dot(x, v)
            if abs(λ - λ_old) < tol
                return λ, x
            end
            λ_old = λ
        end
        error("Power method did not converge in $max_iter iterations.")
    end

    function psd_check(prob::OptProblem, l::AbstractArray{Float64}, tol::Float64=0.0)
        Q = P-> Lq(prob, P, l, 0.0)
        g, v = eig_gpu(Q, length(prob.Ol))
        @show g
        if real(g) < tol
            return v
        end
        return true
    end


    function dual_grad(prob::OptProblem, l::AbstractArray{Float64}, g::Float64, fs::Bool=false)
        Q = P-> Lq(prob, P, l, g)
        L = Ll(prob, l, g)
        P = 0.5 .* (Q \ L)
        n = length(prob.d)
        result = zeros(Float64, 4 * n)
        for i in 1:length(prob.d)
            proj = zeros(ComplexF64, size(prob.Ol))
            proj[prob.d[i]] .= 1.0
            proj = CUDA.Diagonal(CuArray(proj))
            result[i]     = real(prob.Ei' * P) - real(P' * prob.sU(P, proj))
            result[i + n] = imag(prob.Ei' * P) - real(P' * prob.aU(P, proj))
            if fs
                result[i + 2 * n] = real(prob.FS' * P) - real(P' * prob.sU(P, proj))
                result[i + 3 * n] = imag(prob.FS' * P) - real(P' * prob.aU(P, proj))
            end
        end
        return result
    end


    function dual(prob::OptProblem; x0::AbstractArray{Float64}=randn(Float64, 4 * length(prob.d)), max_iter::Int=1000, tol::Float64=1e-9, fs::Bool=false)
        x0[1:length(prob.d)] .= 0.0
	    x0[2 * length(prob.d) + 1:3 * length(prob.d)] .= 0.0
        if !fs
            x0[3 * length(prob.d) + 1:end] .= 0.0
        end
        A = l-> dual_grad(prob, l, 0.0, fs)
        b = zeros(Float64, 4 * length(prob.d))


        n = length(b)
        x = copy(x0)
        xp = copy(x)
        r = b - A(x)
        r0 = copy(r)
        rho = 1.0
        alpha = 1.0
        omega = 1.0
        v = zeros(Float64, n)
        p = zeros(Float64, n)
        
        for iter in 1:max_iter
            psd = psd_check(prob, x)
            if psd != true
                println("Restarting with fake sources")
                prob.FS = psd
                return dual(prob; x0=xp, fs=true)
            end
            @show maximum(abs.(r))

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
            xp = copy(x)
            x = x + alpha * p + omega * s
            r = s - omega * t
        end
        error("BiCGStab did not converge within $max_iter iterations")
     
    end
end