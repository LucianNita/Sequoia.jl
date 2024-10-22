"""
# Example 1: Full Problem Setup with Constraints, Objective, and Initial Guess

This example demonstrates how to set up a `SEQUOIA_pb` problem with constraints, an objective function, and a custom initial guess.

```julia
example_full_problem_setup()

pb = SEQUOIA_pb(
    2, 
    x0 = [0.5, 0.5],               # Initial guess
    is_minimization = true,         # Minimization problem
    solver_settings = SEQUOIA_Settings(:QPM, :LBFGS, false, 1e-6, 1000, 3600.0)  # Solver settings
)

# Define the objective function
objective_fn = x -> sum(x.^2)  # Simple quadratic objective
set_objective!(pb, objective_fn)

# Define constraints
constraints_fn = x -> [x[1] + x[2] - 1.0]  # Constraint: x1 + x2 = 1
set_constraints!(pb, constraints_fn, [1], [])  # Equality constraint

# Print the problem setup
println(pb)

# Expected output:
# SEQUOIA_pb with fields showing the nvar, x0, objective, constraints, and solver settings.
"""

# Example 1: Full Problem Setup with Constraints, Objective, and Initial Guess
function example_full_problem_setup()
    pb = SEQUOIA_pb(
        2, 
        x0 = [0.5, 0.5],               # Initial guess
        is_minimization = true,         # Minimization problem
        solver_settings = SEQUOIA_Settings(:QPM, :LBFGS, false, 1e-6, 1000, 3600.0)  # Solver settings
    )

    # Define the objective function
    objective_fn = x -> sum(x.^2)  # Simple quadratic objective
    set_objective!(pb, objective_fn)

    # Define constraints
    constraints_fn = x -> [x[1] + x[2] - 1.0]  # Constraint: x1 + x2 = 1
    set_constraints!(pb, constraints_fn, [1], Int[])  # Equality constraint

    # Print the problem setup
    println(pb)
end

"""
# Example 2: Unconstrained Minimization Problem with Automatic Differentiation

This example shows how to set up a simple unconstrained minimization problem. The gradient is automatically computed using ForwardDiff.

```julia
example_unconstrained_problem()

pb = SEQUOIA_pb(2)

# Define the objective function
objective_fn = x -> (x[1] - 3.0)^2 + (x[2] - 2.0)^2  # Minimize the distance from (3, 2)
set_objective!(pb, objective_fn)  # No need to set gradient, it will be auto-computed

println(pb)

# Expected output:
# SEQUOIA_pb object with objective set and auto-computed gradient.
"""

# Example 2: Unconstrained Minimization Problem with Automatic Differentiation
function example_unconstrained_problem()
    pb = SEQUOIA_pb(2)

    # Define the objective function
    objective_fn = x -> (x[1] - 3.0)^2 + (x[2] - 2.0)^2  # Minimize the distance from (3, 2)
    set_objective!(pb, objective_fn)  # No need to set gradient, it will be auto-computed

    println(pb)
end

"""
# Example 3: Setting Solver Settings for a Feasibility Problem

This example demonstrates how to set up solver settings for a feasibility problem (no objective). The `set_solver_settings!` function is used to define solver-specific parameters.

```julia
example_feasibility_problem()

pb = SEQUOIA_pb(3)

# Define solver settings for a feasibility problem
settings = SEQUOIA_Settings(
    :QPM,              # Outer method
    :Newton,           # Inner solver
    true,              # Feasibility problem
    1e-6,              # Residual tolerance for constraints
    500,               # Max iterations for outer solver
    1800.0             # Max time for outer solver in seconds
)
set_solver_settings!(pb, settings)

println(pb)

# Expected output:
# SEQUOIA_pb object with solver settings defined for a feasibility problem.
"""

# Example 3: Setting Solver Settings for a Feasibility Problem
function example_feasibility_problem()
    pb = SEQUOIA_pb(3)

    # Define solver settings for a feasibility problem
    settings = SEQUOIA_Settings(
        :QPM,              # Outer method
        :Newton,           # Inner solver
        true,              # Feasibility problem
        1e-6,              # Residual tolerance for constraints
        500,               # Max iterations for outer solver
        1800.0             # Max time for outer solver in seconds
    )
    set_solver_settings!(pb, settings)

    println(pb)
end


"""
# Example 4: Handling an Invalid Objective Function

This example demonstrates the error handling mechanism when an invalid objective function is provided. The `set_objective!` function ensures that the objective is a valid function returning a scalar.

```julia
example_invalid_objective()

pb = SEQUOIA_pb(2)

# Attempt to set an invalid objective function (e.g., returning a vector instead of a scalar)
try
    set_objective!(pb, x -> [x[1] + x[2]])  # This should fail because the output is not a scalar
catch e
    println(e)  # Prints the error: "The objective function must return a scalar of type Float64."
end
"""

# Example 4: Handling an Invalid Objective Function
function example_invalid_objective()
    pb = SEQUOIA_pb(2)

    # Attempt to set an invalid objective function (e.g., returning a vector instead of a scalar)
    try
        set_objective!(pb, x -> [x[1] + x[2]])  # This should fail because the output is not a scalar
    catch e
        println(e)  # Prints the error: "The objective function must return a scalar of type Float64."
    end
end

"""
# Example 5: Setting Constraints and Jacobian with Automatic Differentiation

This example demonstrates how to set constraints and automatically compute the Jacobian using ForwardDiff for the `SEQUOIA_pb` problem.

```julia
example_constraints_and_jacobian()

pb = SEQUOIA_pb(2)

# Define a simple constraint function
constraints_fn = x -> [x[1]^2 + x[2] - 1.0]

# Set the constraints and let the Jacobian be computed automatically
set_constraints!(pb, constraints_fn, eqcon=[1], ineqcon=[])

# Print the problem to verify the constraints and Jacobian setup
println(pb)
"""

# Example 5: Setting Constraints and Jacobian with Automatic Differentiation
function example_constraints_and_jacobian()
    pb = SEQUOIA_pb(2)

    # Define a simple constraint function
    constraints_fn = x -> [x[1]^2 + x[2] - 1.0]

    # Set the constraints and let the Jacobian be computed automatically
    set_constraints!(pb, constraints_fn, eqcon = [1], ineqcon = [])

    # Print the problem to verify the constraints and Jacobian setup
    println(pb)
end

"""
# Example 6: Setting Custom Solver Settings

This example demonstrates how to set custom solver settings for the `SEQUOIA_pb` problem, including changing the optimization method and convergence criteria.

```julia
example_custom_solver_settings()

pb = SEQUOIA_pb(3)

# Define a simple objective function
objective_fn = x -> sum(x.^2)

# Set the objective function for the problem
set_objective!(pb, objective_fn)

# Define custom solver settings
custom_settings = SEQUOIA_Settings(:QPM, :Newton, false, 1e-7, 2000, 5000.0)

# Set custom solver settings for the problem
set_solver_settings!(pb, custom_settings)

# Print the problem to verify the solver settings
println(pb)
"""

# Example 6: Setting Custom Solver Settings
function example_custom_solver_settings()
    pb = SEQUOIA_pb(3)

    # Define a simple objective function
    objective_fn = x -> sum(x.^2)

    # Set the objective function for the problem
    set_objective!(pb, objective_fn)

    # Define custom solver settings
    custom_settings = SEQUOIA_Settings(:QPM, :Newton, false, 1e-7, 2000, 5000.0)

    # Set custom solver settings for the problem
    set_solver_settings!(pb, custom_settings)

    # Print the problem to verify the solver settings
    println(pb)
end

"""
# Example 7: Resetting the Solution History

This example demonstrates how to reset the solution history of a `SEQUOIA_pb` problem instance after solving or modifying the problem setup.

```julia
example_reset_solution_history()

pb = SEQUOIA_pb(2)

# Define an objective function
objective_fn = x -> sum(x.^2)

# Set the objective function for the problem
set_objective!(pb, objective_fn)

# Solve the problem (hypothetical solve, just a placeholder)
println("Solving the problem...")

# Reset the solution history after solving
reset_solution_history!(pb)

# Print the problem to verify that the solution history is reset
println(pb)
"""

# Example 7: Resetting the Solution History
function example_reset_solution_history()
    pb = SEQUOIA_pb(2)

    # Define an objective function
    objective_fn = x -> sum(x.^2)

    # Set the objective function for the problem
    set_objective!(pb, objective_fn)

    # Solve the problem (hypothetical solve, just a placeholder)
    println("Solving the problem...")

    # Reset the solution history after solving
    reset_solution_history!(pb)

    # Print the problem to verify that the solution history is reset
    println(pb)
end

"""
# Example 8: Handling Invalid Exit Codes

This example demonstrates how to update the `exitCode` of a `SEQUOIA_pb` problem instance and how to handle invalid exit codes.

```julia
example_invalid_exit_code()

pb = SEQUOIA_pb(2)

# Attempt to update the exit code with a valid value
update_exit_code!(pb, :OptimalityReached)

# Print the updated problem
println("Updated exit code to :OptimalityReached:")
println(pb)

# Now attempt to update the exit code with an invalid value (this will throw an error)
try
    update_exit_code!(pb, :InvalidCode)
catch e
    println("Caught an error: ", e)
end

Expected output:

Updated exit code to :OptimalityReached:
SEQUOIA_pb(2, ...)
Caught an error: Invalid exit code: `:InvalidCode`. Must be one of: [:NotCalled, :OptimalityReached, :Infeasibility, :MaxIterations, :Unbounded, :SolverError]

""" 

# Example 8: Handling Invalid Exit Codes
function example_invalid_exit_code()
    pb = SEQUOIA_pb(2)

    # Attempt to update the exit code with a valid value
    update_exit_code!(pb, :OptimalityReached)

    # Print the updated problem
    println("Updated exit code to :OptimalityReached:")
    println(pb)

    # Now attempt to update the exit code with an invalid value (this will throw an error)
    try
        update_exit_code!(pb, :InvalidCode)
    catch e
        println("Caught an error: ", e)
    end
end
