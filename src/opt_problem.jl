module opt_problem
    include("../src/utils.jl")
    using .utils, LinearAlgebra, IterativeSolvers, Plots, Random
    export OptProblem, dual, dual_grad
    # Random.seed!(0)


    mutable struct OptProblem
        Ol::AbstractArray{ComplexF64}
        Oq::AbstractArray{ComplexF64}
        d::Vector{Vector{Int}}
        Ei::AbstractArray{ComplexF64}
        U::AbstractArray{ComplexF64}
        FS::AbstractArray{ComplexF64}
    end


    function OptProblem(Ol::AbstractArray{ComplexF64}, Oq::AbstractArray{ComplexF64}, d::Vector{Vector{Int}}, Ei::AbstractArray{ComplexF64}, G::AbstractArray{ComplexF64}, x::ComplexF64)
        X = Diagonal(x * ones(ComplexF64, size(G, 2)))
        U = inv(X)' - G'
        FS = zeros(ComplexF64, length(Ei))
        return OptProblem(Ol, Oq, d, Ei, U, FS)
    end


    function Ll(prob::OptProblem, l::AbstractArray{Float64}, g::Float64)
        n = length(prob.d)
        coef = sum(l[1:n]) + (sum(l[n+1:2*n]) + g) * 1im
        coef_fs = sum(l[2*n+1:3*n]) + sum(l[3*n+1:end]) * 1im
        return prob.Ol + (coef * prob.Ei) + (coef_fs * prob.FS)
    end


    function Lq(prob::OptProblem, l::AbstractArray{Float64}, g::Float64)
        result = prob.Oq + g * asym(prob.U)
        for i in 1:length(prob.d)
            proj = zeros(ComplexF64, size(prob.Ol))
            proj[prob.d[i]] .= 1.0
            proj = Diagonal(proj)
            result +=  sym(prob.U) * proj * l[i]
            result += asym(prob.U) * proj * l[i + length(prob.d)]
            result +=  sym(prob.U) * proj * l[i + 2 * length(prob.d)]
            result += asym(prob.U) * proj * l[i + 3 * length(prob.d)]
        end
        return result
    end


    function partial_constraint(prob::OptProblem, l::AbstractArray{Float64}, g::Float64)
        Q = Lq(prob, l, g)
        L = Ll(prob, l, g)
        P = 0.5 .* (Q \ L)
        return imag(prob.Ei' * P) - real(P' * asym(prob.U) * P)
    end


    function partial_multiplier(prob::OptProblem, l::AbstractArray{Float64})
        Q = Lq(prob, l, 0.0)
        E = asym(prob.U)
        g = -minimum(real.(eigen(Q, E).values))
        if g < 0.0
            return 0.0
        end
        return pade_root(x-> partial_constraint(prob, l, x), g^(5/4))
    end

    function fake_source(prob::OptProblem, l::AbstractArray{Float64}, g::Float64, tol=1e-5)
        e = eigen(Lq(prob, l, g))
        return e.vectors[argmax(abs.(real.(e.values))), :]
    end


    function psd_check(prob::OptProblem, l::AbstractArray{Float64})
        Q = Lq(prob, l, 0.0)
        E = asym(prob.U)
        return minimum(real.(eigen(Q, E).values))
    end


    function dual_grad(prob::OptProblem, l::AbstractArray{Float64}, g::Float64, fs::Bool=false)
        Q = Lq(prob, l, g)
        L = Ll(prob, l, g)
        P = 0.5 .* (Q \ L)
        n = length(prob.d)
        result = zeros(Float64, 4 * n)
        for i in 1:length(prob.d)
            proj = zeros(ComplexF64, size(prob.Ol))
            proj[prob.d[i]] .= 1.0
            proj = Diagonal(proj)
            result[i]     = real(prob.Ei' * P) - real(P' * sym(prob.U * proj) * P)
            result[i + n] = imag(prob.Ei' * P) - real(P' * asym(prob.U * proj) * P)
            if fs
                result[i + 2 * n] = real(prob.FS' * P) - real(P' * sym(prob.U * proj) * P)
                result[i + 3 * n] = imag(prob.FS' * P) - real(P' * asym(prob.U * proj) * P)
            end
        end
        return result
    end


    function dual(prob::OptProblem; x0::AbstractArray{Float64}=randn(Float64, 4*length(prob.d)), tol::Float64=sqrt(eps(Float64)), maxiter::Int64=4*length(prob.d), lr0::Float64 = 1e-3, lr::Float64 = lr0, fs::Bool=false)
        grad = l-> dual_grad(prob, l, 0.0, fs)
        x = copy(x0)
        xp = copy(x)
        g = grad(x)
        gp = copy(g)
        while norm(g) > tol
            @show norm(g), psd_check(prob, x), lr
            if norm(g) < tol
                return x
            end
            if psd_check(prob, x) < 0
                while psd_check(prob, x) < 0
                    println("Backtracking")
                    @show norm(grad(x)), psd_check(prob, x), lr
                    readline()
                    x = xp - lr * g
                    lr /= 2
                end
                prob.FS = fake_source(prob, x, 0.0)
                println("Restarting with fake sources")
                return dual(prob, x0=x, tol=tol, maxiter=maxiter, lr0=lr0, lr=lr, fs=true)
            end
            if norm(g) > norm(gp)
                x = xp
                lr/=2
            end
            if lr < tol
                return x
            end
            xp = copy(x)
            x -= (lr / norm(g)) * g
            gp = copy(g)
            g = grad(x)
        end
        return x
    end
end