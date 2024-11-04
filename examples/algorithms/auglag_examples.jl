#= Example usage: (same as in QPM)

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
solution_history_fixed = alm_solve(obj_fn, grad_fn, cons_fn, cons_jac_fn, lb, ub, eq_indices, ineq_indices, x0, inner_solver, penalty_init=1.0, penalty_mult=10.0, tol=1e-6, max_iter=1000, update_fn=fixed_penalty_update)

println("Solution history for fixed penalty: ", solution_history_fixed)

println("\nUsing adaptive penalty update strategy:")
solution_history_adaptive = alm_solve(obj_fn, grad_fn, cons_fn, cons_jac_fn, lb, ub, eq_indices, ineq_indices, x0, inner_solver, penalty_init=1.0, penalty_mult=10.0, tol=1e-6, max_iter=1000, damping_factor=10.0, update_fn=adaptive_penalty_update)

println("Solution history for adaptive penalty: ", solution_history_adaptive)

# Example of warm starting with Lagrange multipliers
println("\nUsing warm start for Lagrange multipliers:")
solution_history_warm = alm_solve(obj_fn, grad_fn, cons_fn, cons_jac_fn, lb, ub, eq_indices, ineq_indices, x0, inner_solver, penalty_init=1.0, penalty_mult=10.0, tol=1e-6, max_iter=1000, update_fn=fixed_penalty_update, λ_init=final_lambda_fixed)

println("Solution history for warm start: ", solution_history_warm)
=#

using Optim, Sequoia
using LinearAlgebra

# Example 1: Minimizing a quadratic function with an equality constraint
#function example_simple_equality()
    # Objective: Minimize f(x) = (x1 - 2)^2 + (x2 - 3)^2
    objective_fn = x -> (x[1] - 2.0)^2 + (x[2] - 3.0)^2

    # Constraint: x1 + x2 - 5 = 0
    constraints_fn = x -> [x[1] - x[2], x[1] + x[2] - 4.0]

    # Gradient of the objective
    gradient_fn = x -> [2.0 * (x[1] - 2.0), 2.0 * (x[2] - 3.0)]

    # Jacobian of the constraint
    jacobian_fn = x -> [1.0 -1.0; 1.0 1.0] 

    # Initialize SEQUOIA_pb problem
    pb = SEQUOIA_pb(
        2,
        x0 = [0.0, 0.0],                     # Initial guess
        is_minimization = true,               # Minimization problem
        objective = objective_fn,             # Objective function
        gradient = gradient_fn,               # Gradient of the objective
        constraints = constraints_fn,         # Constraints function
        jacobian = jacobian_fn,               # Jacobian of the constraints
        eqcon = [1],                          # Equality constraint index
        ineqcon = [2],                         # No inequality constraints
        solver_settings = SEQUOIA_Settings(:AugLag, :LBFGS, false, 1e-6, 1000, 300),
    )

    # Solve the problem using QPM
    solve!(pb)

    # Display the results
    println("Solution History:")
    for step in pb.solution_history.iterates
        println("Iteration: ", step.outer_iteration_number)
        println("Solution: ", step.x)
        println("Objective Value: ", step.fval)
        println("Constraint Violation: ", step.convergence_metric)
    end