using LinearAlgebra, Optim

# Fixed penalty update function (same as in QPM)
function fixed_penalty_update(penalty_param, penalty_mult)
    return penalty_param * penalty_mult
end

# Adaptive penalty update function (same as in QPM)
function adaptive_penalty_update(penalty_param, constraint_violation, tol, damping_factor)
    return penalty_param * min(max(1, constraint_violation / tol), damping_factor)
end

# Augmented Lagrangian Method (ALM) Implementation using Optim.jl
"""
    alm_solve(obj_fn, grad_fn, cons_fn, cons_jac_fn, lb, ub, eq_indices, ineq_indices, x0, inner_solver;
              penalty_init=1.0, penalty_mult=10.0, tol=1e-6, max_iter=1000, damping_factor=10.0, update_fn=fixed_penalty_update, λ_init=nothing)

Solves a constrained optimization problem using the Augmented Lagrangian Method with `Optim.jl` for solving subproblems.

# Arguments:
- `obj_fn`: The objective function `f(x)` to minimize.
- `grad_fn`: The gradient of the objective function `∇f(x)`.
- `cons_fn`: The constraint function `g(x)`, which returns a vector of constraint values.
- `cons_jac_fn`: The Jacobian of the constraint function `∇g(x)`.
- `lb`: Vector of lower bounds for the constraints.
- `ub`: Vector of upper bounds for the constraints.
- `eq_indices`: Indices specifying which constraints are equality constraints.
- `ineq_indices`: Indices specifying which constraints are inequality constraints.
- `x0`: Initial guess for the solution `x`.
- `inner_solver`: The optimization solver from `Optim.jl` to use for the inner unconstrained subproblem (e.g., `LBFGS()`, `Newton()`, etc.).

# Optional Keyword Arguments:
- `penalty_init`: Initial penalty parameter (default = 1.0).
- `penalty_mult`: Multiplicative factor for penalty parameter updates (default = 10.0).
- `tol`: Tolerance for convergence (default = 1e-6).
- `max_iter`: Maximum number of iterations (default = 1000).
- `damping_factor`: A factor to limit the rate of penalty increase for adaptive updates (default = 10.0).
- `update_fn`: Penalty update function to use, either `fixed_penalty_update` or `adaptive_penalty_update`.
- `λ_init`: Initial values for the Lagrange multipliers (default = `nothing`, which initializes `λ` to zeros).

# Returns:
- `x`: Solution vector `x` that minimizes the objective function subject to the constraints.
- `penalty_param`: Final penalty parameter.
- `λ`: Final Lagrange multipliers.
"""
function alm_solve(obj_fn, grad_fn, cons_fn, cons_jac_fn, lb, ub, eq_indices, ineq_indices, x0, inner_solver;
                   penalty_init=1.0, penalty_mult=10.0, tol=1e-6, max_iter=1000, damping_factor=10.0, update_fn=fixed_penalty_update, λ_init=zeros(length(lb)))

    # Initialize variables
    x = copy(x0)
    penalty_param = penalty_init
    λ = copy(λ_init)  # Initialize or warm-start Lagrange multipliers
    iteration = 0

    # Helper function: Augmented Lagrangian objective
    function augmented_lagrangian(x, penalty_param, λ)
        constraint_val = cons_fn(x)
        eq_penalty_term = 0.0
        ineq_penalty_term = 0.0

        # Handle equality constraints with Lagrange multipliers
        for i in eq_indices
            eq_penalty_term += λ[i] * (constraint_val[i] - lb[i]) + 0.5 * penalty_param * (constraint_val[i] - lb[i])^2
        end

        # Handle inequality constraints
        for i in ineq_indices
            if constraint_val[i] < lb[i]
                ineq_penalty_term += λ[i] * (lb[i] - constraint_val[i]) + 0.5 * penalty_param * (lb[i] - constraint_val[i])^2
            elseif constraint_val[i] > ub[i]
                ineq_penalty_term += λ[i] * (constraint_val[i] - ub[i]) + 0.5 * penalty_param * (constraint_val[i] - ub[i])^2
            end
        end

        # Return the augmented Lagrangian objective
        return obj_fn(x) + eq_penalty_term + ineq_penalty_term
    end

    # Gradient of the augmented Lagrangian
    function augmented_lagrangian_gradient!(grad_storage, x, penalty_param, λ)
        constraint_val = cons_fn(x)
        grad_obj = grad_fn(x)
        
        grad_eq_penalty = zeros(length(x))
        grad_ineq_penalty = zeros(length(x))

        # Handle equality constraints
        for i in eq_indices
            grad_eq_penalty += (λ[i] + penalty_param * (constraint_val[i] - lb[i])) * cons_jac_fn(x)[:, i]
        end

        # Handle inequality constraints
        for i in ineq_indices
            if constraint_val[i] < lb[i]
                grad_ineq_penalty += (λ[i] + penalty_param * (lb[i] - constraint_val[i])) * cons_jac_fn(x)[:, i]
            elseif constraint_val[i] > ub[i]
                grad_ineq_penalty += (λ[i] + penalty_param * (constraint_val[i] - ub[i])) * cons_jac_fn(x)[:, i]
            end
        end

        grad_storage .= grad_obj .+ grad_eq_penalty .+ grad_ineq_penalty
    end

    # Function to compute the total constraint violation
    function compute_constraint_violation(x)
        constraint_val = cons_fn(x)
        eq_violation = norm(constraint_val[eq_indices] .- lb[eq_indices])
        ineq_violation = norm(max.(constraint_val[ineq_indices] .- ub[ineq_indices], lb[ineq_indices] .- constraint_val[ineq_indices]))
        return eq_violation + ineq_violation
    end

    while iteration < max_iter
        # Define the augmented Lagrangian function
        obj_aug_fn = x -> augmented_lagrangian(x, penalty_param, λ)
        grad_aug_fn! = (g, x) -> augmented_lagrangian_gradient!(g, x, penalty_param, λ)

        # Optim options
        options = Optim.Options(g_tol=tol, iterations=10000, show_trace=false)

        # Solve the unconstrained subproblem
        result = optimize(obj_aug_fn, grad_aug_fn!, x, inner_solver, options)
        x = Optim.minimizer(result)

        # Compute constraint violation
        constraint_violation = compute_constraint_violation(x)

        # Check for convergence
        if constraint_violation < tol
            println("Converged after $iteration iterations.")
            return x, penalty_param, λ
        end

        # Update Lagrange multipliers
        constraint_val = cons_fn(x)
        for i in eq_indices
            λ[i] += penalty_param * (constraint_val[i] - lb[i])
        end
        for i in ineq_indices
            if constraint_val[i] < lb[i]
                λ[i] += penalty_param * (lb[i] - constraint_val[i])
            elseif constraint_val[i] > ub[i]
                λ[i] += penalty_param * (constraint_val[i] - ub[i])
            end
        end

        # Update penalty parameter
        if update_fn === fixed_penalty_update
            penalty_param = update_fn(penalty_param, penalty_mult)
        else
            penalty_param = update_fn(penalty_param, constraint_violation, tol, damping_factor)
        end

        iteration += 1
    end

    println("Maximum iterations reached.")
    return x, penalty_param, λ
end

# Example usage: (same as in QPM)

# Objective function: minimize f(x) = x₁² + x₂²
obj_fn = x -> x[1]^2 + x[2]^2

# Gradient of the objective function
grad_fn = x -> [2*x[1], 2*x[2]]

# Constraint function g(x) = [x₁ + x₂, x₁]
cons_fn = x -> [x[1] + x[2], x[1]]

# Jacobian of the constraint function
cons_jac_fn = x -> [1.0 1.0; 1.0 0.0]

# Lower and upper bounds for constraints
lb = [1.0, -Inf]
ub = [1.0, 0.3]

# Indices for equality and inequality constraints
eq_indices = [1]
ineq_indices = [2]

# Initial guess
x0 = [0.25, 0.75]

# Solve using Augmented Lagrangian Method with Optim.jl
inner_solver = Optim.LBFGS()

println("Using fixed penalty update strategy:")
x_opt_fixed, final_penalty_fixed, final_lambda_fixed = alm_solve(obj_fn, grad_fn, cons_fn, cons_jac_fn, lb, ub, eq_indices, ineq_indices, x0, inner_solver, penalty_init=1.0, penalty_mult=10.0, tol=1e-6, max_iter=1000, update_fn=fixed_penalty_update)

println("Optimal solution (fixed penalty): ", x_opt_fixed)
println("Final penalty parameter (fixed penalty): ", final_penalty_fixed)
println("Final Lagrange multipliers (fixed penalty): ", final_lambda_fixed)

println("\nUsing adaptive penalty update strategy:")
x_opt_adaptive, final_penalty_adaptive, final_lambda_adaptive = alm_solve(obj_fn, grad_fn, cons_fn, cons_jac_fn, lb, ub, eq_indices, ineq_indices, x0, inner_solver, penalty_init=1.0, penalty_mult=10.0, tol=1e-6, max_iter=1000, damping_factor=10.0, update_fn=adaptive_penalty_update)

println("Optimal solution (adaptive penalty): ", x_opt_adaptive)
println("Final penalty parameter (adaptive penalty): ", final_penalty_adaptive)
println("Final Lagrange multipliers (adaptive penalty): ", final_lambda_adaptive)

# Example of warm starting with Lagrange multipliers
println("\nUsing warm start for Lagrange multipliers:")
x_opt_warm, final_penalty_warm, final_lambda_warm = alm_solve(obj_fn, grad_fn, cons_fn, cons_jac_fn, lb, ub, eq_indices, ineq_indices, x0, inner_solver, penalty_init=1.0, penalty_mult=10.0, tol=1e-6, max_iter=1000, update_fn=fixed_penalty_update, λ_init=final_lambda_fixed)

println("Optimal solution (warm start): ", x_opt_warm)
println("Final penalty parameter (warm start): ", final_penalty_warm)
println("Final Lagrange multipliers (warm start): ", final_lambda_warm)
