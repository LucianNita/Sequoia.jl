using LinearAlgebra
using Optim
using Plots

### Problem Definition

# Define the objective function
function f(x::Vector)
    return x[1]^2 + x[2]^2
end

# Equality constraint: x1 + x2 - 1 = 0
function g(x::Vector)
    return x[1] + x[2] - 1
end

# Struct to store the problem
struct SimpleProblem
    f::Function
    g::Function
end

# Define the problem
problem = SimpleProblem(f, g)

### Augmented Lagrangian Function

# Define the augmented Lagrangian function with penalty and Lagrange multiplier
function augmented_lagrangian(x, μ, λ, problem::SimpleProblem)
    # Augmented Lagrangian: objective + penalty and multiplier terms for constraint
    return problem.f(x) + λ * problem.g(x) + 0.5 * μ * problem.g(x)^2
end

# Hessian of the augmented Lagrangian function
function augmented_lagrangian_hessian(x, μ, problem::SimpleProblem)
    # Hessian of the objective function f(x) = x1^2 + x2^2
    H_f = [2.0 0.0; 0.0 2.0]
    
    # Gradient of the constraint g(x) = x1 + x2 - 1
    grad_g = [1.0, 1.0]
    
    # Hessian of the augmented term μ * g(x)^2 / 2
    H_penalty = μ * (grad_g * grad_g')  # Outer product
    
    # Total Hessian
    return H_f + H_penalty
end

### Main Solver Loop for Condition Number Analysis
function run()
# Initial conditions
x0 = [0.5, 0.5]         # Starting point for x
λ = 0.0                  # Initial Lagrange multiplier
penalty_params = [0.1, 1, 10, 100, 1000]  # Penalty parameters to analyze
tol = 1e-5               # Convergence tolerance
max_iter = 50            # Maximum iterations

# Arrays to store results
condition_numbers = []
solutions = []

# Loop over penalty parameters
for μ in penalty_params
    # Define the augmented Lagrangian function and optimizer options
    aug_lagrangian = x -> augmented_lagrangian(x, μ, λ, problem)
    result = optimize(aug_lagrangian, x0, BFGS())

    # Extract optimal x and compute condition number of the Hessian
    x_opt = Optim.minimizer(result)
    H = augmented_lagrangian_hessian(x_opt, μ, problem)
    cond_num = cond(H)

    # Store results
    push!(condition_numbers, cond_num)
    push!(solutions, x_opt)

    # Update Lagrange multiplier based on constraint violation
    λ += μ * g(x_opt)
end

### Plot Condition Number vs Penalty Parameter

plot(penalty_params, condition_numbers, xlabel="Penalty Parameter (μ, log scale)", 
     ylabel="Condition Number (log scale)", 
     title="Condition Number of Hessian vs Penalty Parameter in Augmented Lagrangian",
     xscale=:log10, yscale=:log10, marker=:o, lw=2)
end