using LinearAlgebra
using Optim
using Plots

# Original objective function
function objective(x::Vector)
    return x[1]^2 + x[2]^2
end

# Constraint function (x + y - 1 = 0)
function constraint(x::Vector)
    return x[1] + x[2] - 1
end

# Quadratic penalty function
function penalized_objective(x::Vector, penalty_param)
    return objective(x) + penalty_param * constraint(x)^2
end

# Hessian of the penalized objective function
function penalized_hessian(x::Vector, penalty_param)
    H_obj = [2.0 0.0; 0.0 2.0]  # Hessian of the objective function (x^2 + y^2)
    grad_constr = [1.0, 1.0]     # Gradient of the constraint (x + y - 1)
    H_penalty = 2 * penalty_param * (grad_constr * grad_constr')  # Hessian of the penalty term
    return H_obj + H_penalty
end

# Solve the quadratic penalty problem for a series of penalty parameters
penalty_params = [0.1, 1, 10, 100, 1000]  # Adjust these as needed
condition_numbers = []

for penalty_param in penalty_params
    # Define the penalized objective function for the current penalty parameter
    penalized_func(x) = penalized_objective(x, penalty_param)
    
    # Solve the unconstrained optimization problem
    result = optimize(penalized_func, [0.0, 0.0], BFGS())
    optimal_x = Optim.minimizer(result)
    
    # Compute the condition number of the Hessian at the solution point
    H = penalized_hessian(optimal_x, penalty_param)
    cond_number = cond(H)  # Condition number of the Hessian matrix
    push!(condition_numbers, cond_number)
end

# Plot Condition Number vs Penalty Parameter
plot(penalty_params, condition_numbers, 
     xlabel="Penalty Parameter (log scale)", 
     ylabel="Condition Number (log scale)", 
     title="Condition Number vs Penalty Parameter", 
     xscale=:log10, 
     yscale=:log10, 
     lw=2, marker=:o)
