using LinearAlgebra, JLD2

function bicgstab(A::AbstractArray{ComplexF64}, b::AbstractArray{ComplexF64}, x0::AbstractArray{ComplexF64}=randn(ComplexF64, size(A, 2)), tol::Float64=sqrt(eps(Float64)), maxiter::Int64=size(A, 2))
    x = copy(x0)
    r = b - A * x
    r_hat = copy(r)
    rho = dot(r_hat, r)
    p = r
    for i in 1:maxiter
        v = A * p
        alpha = rho / dot(r_hat, v)
        h = x + alpha * p
        s = r - alpha * v
        if norm(s) < tol
            return x
        end

        t = A * s
        omega = dot(t, s) / dot(t, t)
        x = h + omega * s
        r = s - omega * t
        if norm(r) < tol
            return x
        end
        rho_old = copy(rho)
        rho = dot(r_hat, r)        
        beta = (rho / rho_old) * (alpha / omega)
        p = r + beta * (p - omega * v)
    end
end


sym = M::AbstractMatrix -> 0.5 * (M + M')
asym = M::AbstractMatrix -> -0.5im * (M - M')


n = 3 * (2^3)
A = jldopen("data/greens_matrix_cpu_2.jld2")["result"]
A = sym(Diagonal(ones(ComplexF64, n)) - A')
b = rand(ComplexF64, n)
x = bicgstab(A, b)
@show norm(A * x - b)