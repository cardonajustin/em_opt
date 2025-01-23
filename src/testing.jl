using LinearAlgebra, CUDA, SparseArrays, IterativeSolvers

function bicgstab(A, b::CuArray{ComplexF64}; tol=eps(Float64), max_iter=length(b))
    n = length(b)
    x = CUDA.zeros(Float64, n)
    r = b - A(x)
    r_hat = copy(r)
    rho_old = 1.0
    alpha = 1.0
    omega_old = 1.0
    v = CUDA.zeros(Float64, n)
    p = CUDA.zeros(Float64, n)
    tol2 = tol^2

    for iter in 1:max_iter
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
        real(dot(s, s)) > tol2 || return x + alpha * p, iter

        t = A(s)
        omega_new = dot(t, s) / dot(t, t)
        x += alpha * p + omega_new * s
        r = s - omega_new * t
        real(dot(r, r)) > tol2 || return x, iter        

        rho_old = rho_new
        omega_old = omega_new
    end
    error("Max iterations reached")
end


# Define a test system
function random_well_conditioned_matrix(n, cond_number)
    # Generate random orthogonal matrices U and V
    U, _ = qr(randn(n, n))
    V, _ = qr(randn(n, n))
    
    # Create singular values to achieve the desired condition number
    singular_values = range(1.0, stop=cond_number, length=n)
    
    # Construct the matrix
    A = U * Diagonal(singular_values) * V'
    return A
end
n=100
A = CuArray(random_well_conditioned_matrix(n, 10) * 1e-1 + I) 
b = CUDA.rand(ComplexF64, n)


# Solve using BiCGSTAB
x, iter = bicgstab(x-> A * x, b)

# Display the result
println("Residual: ", norm(A * x - b))
println("Iterations: ", iter)