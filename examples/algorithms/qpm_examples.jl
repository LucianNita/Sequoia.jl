#=
# Example usage: (for testing)

# Define a simple objective function: minimize f(x) = x₁² + x₂²
obj_fn = x -> x[1]^2 + x[2]^2
# Gradient of the objective function
grad_fn = x -> [2*x[1], 2*x[2]]
# Define a constraint function g(x) = [x₁ + x₂, x₁]
cons_fn = x -> [x[1] + x[2]-1, x[1]-0.3]
# Jacobian of the constraint function
cons_jac_fn = x -> [1.0 1.0; 1.0 0.0]

# Indices for equality and inequality constraints
eq_indices = [1]
ineq_indices = [2]

# Initial guess
x0 = [0.25, 0.75]

# Solve using Quadratic Penalty Method with Optim.jl and the chosen inner solver
inner_solver = Optim.LBFGS()

=#

using Optim, Sequoia
using LinearAlgebra

# Example 1: Minimizing a quadratic function with an equality constraint
#function example_simple_equality()
    # Objective: Minimize f(x) = (x1 - 2)^2 + (x2 - 3)^2
    objective_fn = x -> (x[1] - 10.0)^2 + (x[2] - 2.0)^2

    # Constraint: x1 + x2 - 5 = 0
    constraints_fn = x -> [x[1] + x[2] - 5.0]

    # Gradient of the objective
    gradient_fn = x -> [2.0 * (x[1] - 10.0), 2.0 * (x[2] - 2.0)]

    # Jacobian of the constraint
    jacobian_fn = x -> [1.0 1.0]

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
        ineqcon = Int[],                         # No inequality constraints
        solver_settings = SEQUOIA_Settings(:QPM, :LBFGS, false, 1e-6, 1000, 300),
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
#end

# Call the example
#example_simple_equality()
