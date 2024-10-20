using Optim
using LinearAlgebra

# Fixed penalty update function
"""
    fixed_penalty_update(penalty_param, penalty_mult)

Updates the penalty parameter by a fixed multiplicative factor.

# Arguments:
- `penalty_param`: The current penalty parameter.
- `penalty_mult`: The multiplicative factor to update the penalty.

# Returns:
- The updated penalty parameter.
"""
function fixed_penalty_update(penalty_param, penalty_mult)
    return penalty_param * penalty_mult  # Fixed penalty update (simple multiplication)
end


# Adaptive penalty update function
"""
    adaptive_penalty_update(penalty_param, constraint_violation, tol, damping_factor)

Updates the penalty parameter adaptively based on the magnitude of the constraint violation.

# Arguments:
- `penalty_param`: The current penalty parameter.
- `constraint_violation`: The computed violation of the constraints.
- `tol`: The tolerance threshold for constraint satisfaction.
- `damping_factor`: A factor to limit how rapidly the penalty parameter can grow.

# Returns:
- The updated penalty parameter.
"""
function adaptive_penalty_update(penalty_param, constraint_violation, tol, damping_factor)
    # Penalty increases faster when constraint violations are large, slower when violations are small
    return penalty_param * min(max(1, constraint_violation / tol), damping_factor)
end

# Quadratic Penalty Method (QPM) Implementation using Optim.jl
"""
    qpm_solve(obj_fn, grad_fn, cons_fn, cons_jac_fn, lb, ub, eq_indices, ineq_indices, x0, inner_solver;
              penalty_init=1.0, penalty_mult=10.0, tol=1e-6, max_iter=1000, damping_factor=10.0, update_fn=fixed_penalty_update)

Solves a constrained optimization problem using the Quadratic Penalty Method with `Optim.jl` for solving subproblems.

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

# Returns:
- `x`: Solution vector `x` that minimizes the objective function subject to the constraints.
- `penalty_param`: Final penalty parameter.
"""
function qpm_solve(obj_fn, grad_fn, cons_fn, cons_jac_fn, lb, ub, eq_indices, ineq_indices, x0, inner_solver;
                   penalty_init=1.0, penalty_mult=10.0, tol=1e-6, max_iter=1000, damping_factor=10.0, update_fn=fixed_penalty_update)

    # Verify input consistency
    verify_input_consistency(cons_fn, x0, lb, ub, eq_indices, ineq_indices)

    # Initialize variables
    x = copy(x0)  # Copy the initial guess to avoid modifying the original
    penalty_param = penalty_init  # Set initial penalty parameter
    iteration = 0  # Initialize iteration counter

    # Helper function: Augmented objective with penalty for equality and inequality constraints
    function augmented_objective(x, penalty_param)
        constraint_val = cons_fn(x)
        eq_penalty_term = 0.0
        ineq_penalty_term = 0.0

        # Handle equality constraints (penalty applied if violated)
        for i in eq_indices
            eq_penalty_term += 0.5 * penalty_param * (constraint_val[i] - lb[i])^2
        end

        # Handle inequality constraints (penalty applied if outside bounds)
        for i in ineq_indices
            if constraint_val[i] < lb[i]
                ineq_penalty_term += 0.5 * penalty_param * (lb[i] - constraint_val[i])^2
            elseif constraint_val[i] > ub[i]
                ineq_penalty_term += 0.5 * penalty_param * (constraint_val[i] - ub[i])^2
            end
        end

        # Return the augmented objective (original objective + penalties)
        return obj_fn(x) + eq_penalty_term + ineq_penalty_term
    end

    # Helper function: In-place gradient of the augmented objective with penalty
    function augmented_gradient!(grad_storage, x, penalty_param)
        constraint_val = cons_fn(x)
        grad_obj = grad_fn(x)
        
        # Initialize penalty gradients
        grad_eq_penalty = zeros(length(x))
        grad_ineq_penalty = zeros(length(x))

        # Handle equality constraints
        for i in eq_indices
            grad_eq_penalty += penalty_param * cons_jac_fn(x)[:, i] * (constraint_val[i] - lb[i])
        end

        # Handle inequality constraints
        for i in ineq_indices
            if constraint_val[i] < lb[i]
                grad_ineq_penalty += penalty_param * cons_jac_fn(x)[:, i] * (lb[i] - constraint_val[i])
            elseif constraint_val[i] > ub[i]
                grad_ineq_penalty += penalty_param * cons_jac_fn(x)[:, i] * (constraint_val[i] - ub[i])
            end
        end

        # Update the gradient storage with the objective gradient + penalties
        grad_storage .= grad_obj .+ grad_eq_penalty .+ grad_ineq_penalty
    end

    # Function to compute the total constraint violation (used for adaptive penalty updates)
    function compute_constraint_violation(x)
        constraint_val = cons_fn(x)
        eq_violation = norm(constraint_val[eq_indices] .- lb[eq_indices])  # Equality violation
        ineq_violation = norm(max.(constraint_val[ineq_indices] .- ub[ineq_indices], lb[ineq_indices] .- constraint_val[ineq_indices]))  # Inequality violation
        return eq_violation + ineq_violation
    end

    while iteration < max_iter
        # Use Optim.jl to minimize the augmented objective function
        obj_aug_fn = x -> augmented_objective(x, penalty_param)
        grad_aug_fn! = (g, x) -> augmented_gradient!(g, x, penalty_param)
        
        # Set Optim options
        options = Optim.Options(g_tol=tol, iterations=10000, show_trace=false)

        # Solve the unconstrained subproblem using the inner solver
        result = optimize(obj_aug_fn, grad_aug_fn!, x, inner_solver, options)

        # Extract the optimized solution from the subproblem
        x = Optim.minimizer(result)
        
        # Compute the constraint violation for adaptive updates
        constraint_violation = compute_constraint_violation(x)

        # Check for convergence based on the constraint violation
        if constraint_violation < tol
            println("Converged after $iteration iterations.")
            return x_opt, penalty_param, 0.01, num_inner_iterations, solver_status
        end

        # Check the number of arguments required by the update function
        if update_fn === fixed_penalty_update
            # Call fixed penalty update function (with two arguments)
            penalty_param = update_fn(penalty_param, penalty_mult)
        else
            # Call adaptive penalty update function (with four arguments)
            penalty_param = update_fn(penalty_param, constraint_violation, tol, damping_factor)
        end
        
        # Increment the iteration counter
        iteration += 1
    end

    # If maximum iterations are reached, return the current solution and penalty
    println("Maximum iterations reached.")
    return x_opt, penalty_param, 0.01, num_inner_iterations, solver_status
end

# Example usage: (for testing)

# Define a simple objective function: minimize f(x) = x₁² + x₂²
obj_fn = x -> x[1]^2 + x[2]^2

# Gradient of the objective function
grad_fn = x -> [2*x[1], 2*x[2]]

# Define a constraint function g(x) = [x₁ + x₂, x₁]
cons_fn = x -> [x[1] + x[2], x[1]]

# Jacobian of the constraint function
cons_jac_fn = x -> [1.0 1.0; 1.0 0.0]

# Lower and upper bounds for constraints
lb = [1.0, -Inf]   # First constraint is equality, second is an inequality
ub = [1.0, 0.3]

# Indices for equality and inequality constraints
eq_indices = [1]
ineq_indices = [2]

# Initial guess
x0 = [0.25, 0.75]

# Solve using Quadratic Penalty Method with Optim.jl and the chosen inner solver
inner_solver = Optim.LBFGS()

# Use fixed penalty update strategy
println("Using fixed penalty update strategy:")
x_opt_fixed, final_penalty_fixed = qpm_solve(obj_fn, grad_fn, cons_fn, cons_jac_fn, lb, ub, eq_indices, ineq_indices, x0, inner_solver, penalty_init=1.0, penalty_mult=10.0, tol=1e-6, max_iter=1000, update_fn=fixed_penalty_update)

println("Optimal solution (fixed penalty): ", x_opt_fixed)
println("Final penalty parameter (fixed penalty): ", final_penalty_fixed)

# Use adaptive penalty update strategy
println("\nUsing adaptive penalty update strategy:")
x_opt_adaptive, final_penalty_adaptive = qpm_solve(obj_fn, grad_fn, cons_fn, cons_jac_fn, lb, ub, eq_indices, ineq_indices, x0, inner_solver, penalty_init=1.0, penalty_mult=10.0, tol=1e-6, max_iter=1000, damping_factor=10.0, update_fn=adaptive_penalty_update)

println("Optimal solution (adaptive penalty): ", x_opt_adaptive)
println("Final penalty parameter (adaptive penalty): ", final_penalty_adaptive)
