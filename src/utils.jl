module utils
    using GilaElectromagnetics, LinearAlgebra, JLD2, BaryRational
    export sym, asym, greens_matrix, pade_root


    sym  = M::AbstractMatrix ->  0.5   * (M + M')
    asym = M::AbstractMatrix -> -0.5im * (M - M')


    function greens_matrix(cells::NTuple{3, Int}, scale::NTuple{3, Rational{Int}}, name::String)
        operator = GlaOpr(cells, scale; setTyp=ComplexF64, useGpu=false)
        result = zeros(ComplexF64, size(operator, 1), size(operator, 2))
        for i in 1:prod(cells)
            v = zeros(ComplexF64, size(operator, 2))
            v[i] = 1.0
            result[:, i] = operator * v
        end
        jldsave(name; result)        
        return result
    end


    function pade_root(f, z_init::Float64; max_iter::Int=5, max_restart::Int=5, r::Float64=1.0, tol=sqrt(eps(Float64)))
        err = 0
        for _ in 0:max_restart
            err_init = f(z_init)
            z = z_init + sign(err_init) * min(abs(err_init), r)
            err = f(z)
            domain = [z, z_init]
            codomain = [err, err_init]
            for iter in 1:max_iter
                a = aaa(domain, codomain, clean=1)
                poles, _, zeros = prz(a)
                z = maximum(real.(zeros))
                err = f(z)
                abs(err) > tol || return z
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