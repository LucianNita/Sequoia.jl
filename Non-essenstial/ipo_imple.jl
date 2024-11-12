using ForwardDiff
using LinearAlgebra

# Define the nonlinear optimization problem structure
struct Problem
    f::Function                 # Objective function f(x)
    grad_f::Function             # Gradient of the objective function
    h::Function                  # Equality constraints h(x)
    grad_h::Function             # Gradient of equality constraints
    g::Function                  # Inequality constraints g(x)
    grad_g::Function             # Gradient of inequality constraints
end

# KKT system for the barrier problem
function kkt_system(problem::Problem, x, s, λ, η, μ)
    # Evaluate functions and gradients at current x and s
    f_grad = problem.grad_f(x)
    h_vals = problem.h(x)
    g_vals = problem.g(x)
    h_grads = problem.grad_h(x)
    g_grads = problem.grad_g(x)

    # Stationarity conditions
    stationarity_x = f_grad + h_grads' * λ + g_grads' * η
    stationarity_s = -μ ./ s .+ 2 .* η .* s 

    # Primal feasibility conditions
    primal_feasibility_h = h_vals
    primal_feasibility_g = g_vals .+ s .^ 2

    # Dual feasibility (η >= 0 is implicitly enforced by barrier)
    
    # Stack all KKT conditions
    return vcat(stationarity_x, stationarity_s, primal_feasibility_h, primal_feasibility_g)
end


function solve_kkt_system(problem::Problem, x, s, λ, η, μ; tol = 1e-6, max_iter = 100)
    for iter in 1:max_iter
        # Compute gradients and function values using ForwardDiff
        f_grad = problem.grad_f(x)
        h_vals = problem.h(x)
        g_vals = problem.g(x)
        
        # Compute Jacobians of the constraints using ForwardDiff
        h_grads = ForwardDiff.jacobian(problem.h, x)  # Jacobian of h(x) w.r.t x
        g_grads = ForwardDiff.jacobian(problem.g, x)  # Jacobian of g(x) w.r.t x
        
        # Compute the residuals for the KKT conditions
        stationarity_x = f_grad + h_grads' * λ + g_grads' * η
        stationarity_s = -μ ./ s .+ 2 .* η .* s
        primal_feasibility_h = h_vals
        primal_feasibility_g = g_vals .+ s .^ 2

        # Construct the KKT residual vector
        residual = vcat(stationarity_x, primal_feasibility_h, primal_feasibility_g, stationarity_s)
        
        # Check convergence
        if maximum(abs.(residual)) < tol
            break
        end

        # Construct the KKT matrix with ForwardDiff Hessians
        n_x = length(x)
        n_s = length(s)
        n_h = length(h_vals)
        n_g = length(g_vals)

        # Compute Hessian of the Lagrangian using ForwardDiff
        hessian_f = ForwardDiff.hessian(problem.f, x)  # Hessian of f(x)
        
        # Initialize the Hessian of the Lagrangian
        hessian_L = hessian_f
        for i in 1:n_g
            hessian_g_i = ForwardDiff.hessian(z -> problem.g(z)[i], x)
            hessian_L += η[i] * hessian_g_i
        end

        # Diagonal matrices
        D_s = Diagonal(2 * s)
        D_mu_s2 = Diagonal(-μ ./ s .^ 2)

        # Build KKT matrix
        KKT_matrix = [
            hessian_L             h_grads'       g_grads'        zeros(n_x, n_s);
            h_grads               zeros(n_h, n_h) zeros(n_h, n_g) zeros(n_h, n_s);
            g_grads               zeros(n_g, n_h) zeros(n_g, n_g) D_s;
            zeros(n_s, n_x)       zeros(n_s, n_h) D_s             D_mu_s2
        ]
        
        # Solve for Newton direction
        Δ = -KKT_matrix \ residual
        
        # Extract increments for each variable
        Δx = Δ[1:n_x]
        Δλ = Δ[n_x+1:n_x+n_h]
        Δη = Δ[n_x+n_h+1:n_x+n_h+n_g]
        Δs = Δ[n_x+n_h+n_g+1:end]

        # Update variables
        x += Δx
        λ += Δλ
        η += Δη
        s += Δs
        s = max.(s, ε)
    end
    
    return x, s
end


# Main function for the interior point method
function interior_point_method(problem::Problem, x_init, s_init, λ_init, η_init; μ_init = 1.0, tol = 1e-6, max_iter = 100)
    x, s, λ, η, μ = x_init, s_init, λ_init, η_init, μ_init
    for iter in 1:max_iter
        x, s = solve_kkt_system(problem, x, s, λ, η, μ; tol = tol)
        
        # Decrease barrier parameter
        # Dynamically adjust barrier parameter based on the minimum value of s
        μ = min(μ * 0.1, μ * minimum(s) / ε)
        if μ < tol
            break
        end
    end
    return x
end

# Define the problem functions and their gradients
objective(x) = (x[1]-1)^2 + (x[2]-2)^2
grad_objective(x) = [2 * (x[1]-1), 2 * (x[2]-2)]

equality_constraint(x) = [x[1] + x[2] - 3]
grad_equality_constraint(x) = [1 1]  # Row vector

inequality_constraint(x) = [-x[1]^2 + x[2]]
grad_inequality_constraint(x) = [-2 * x[1] 1]  # Row vector for gradient of g

# Set up the problem in the Problem struct
problem = Problem(
    objective,
    grad_objective,
    equality_constraint,
    grad_equality_constraint,
    inequality_constraint,
    grad_inequality_constraint
)
ε = 1e-8  # Small threshold for slack variable
# Initial guesses for x, s, λ, η
x_init = [0.1, 0.2]
s_init = [1.0]  # Initial guess for slack variable
λ_init = [0.0]  # Initial guess for Lagrange multiplier of equality constraint
η_init = [0.0]  # Initial guess for Lagrange multiplier of inequality constraint

# Call the interior point method
solution = interior_point_method(problem, x_init, s_init, λ_init, η_init; μ_init = 1.0, tol = 1e-6, max_iter = 100)

# Display solution
println("Optimal solution for x: ", solution)
println("Objective value at optimal solution: ", objective(solution))
