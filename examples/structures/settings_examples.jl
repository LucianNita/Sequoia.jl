"""
# SEQUOIA_Settings Examples

This file contains example use cases for the `SEQUOIA_Settings` struct, demonstrating its usage in different scenarios:
1. Full constructor usage with all parameters specified.
2. Minimal constructor usage with default values for optional fields.
3. Error handling with invalid inputs.
4. Custom solver parameters.
"""

# Example 1: Using the Full Constructor
"""
This example shows how to use the full constructor for `SEQUOIA_Settings`, where all fields are specified, 
including inner and outer solvers, convergence criteria, tolerances, and optional parameters.

# Usage:
    example_full_constructor()

# Expected Output:
    SEQUOIA_Settings(:SEQUOIA, :LBFGS, false, 1.0e-8, 1000, 3600.0, 1.0e-5, :GradientNorm, 500, 300.0, true, 0.0001, -1.0e6, 1.0e-8, [1.0, 0.5])
"""
function example_full_constructor()
    settings_full = SEQUOIA_Settings(
        :SEQUOIA,          # Outer method
        :LBFGS,            # Inner solver
        false,             # Feasibility: solving an optimization problem
        1e-8,              # Residual tolerance for constraints
        1000,              # Max iterations for outer solver
        3600.0,            # Max time for outer solver in seconds
        1e-5,              # Gradient norm tolerance
        conv_crit = :GradientNorm, # Convergence criterion
        max_iter_inner = 500,      # Max inner iterations
        max_time_inner = 300.0,    # Max time for inner solver
        store_trace = true,        # Enable tracing
        cost_tolerance = 1e-4,     # Desired optimality gap
        cost_min = -1e6,           # Minimum cost
        step_min = 1e-8,           # Minimum step size
        solver_params = [1.0, 0.5] # Solver-specific parameters
    )
    println(settings_full)
end

# Example 2: Handling Missing Optional Parameters
"""
This example demonstrates the minimal constructor for `SEQUOIA_Settings`, which defaults some fields such as
convergence criteria (`:GradientNorm`) and omits optional parameters like `cost_tolerance` and `solver_params`.

# Usage:
    example_minimal_constructor()

# Expected Output:
    SEQUOIA_Settings(:QPM, :Newton, true, 1.0e-6, 500, 1800.0, 1.0e-6, :GradientNorm, nothing, nothing, false, nothing, nothing, nothing, nothing)
"""
function example_minimal_constructor()
    settings_min = SEQUOIA_Settings(
        :QPM,              # Outer method
        :Newton,           # Inner solver
        true,              # Feasibility: solving a feasibility problem
        1e-6,              # Residual tolerance for constraints
        500,               # Max iterations for outer solver
        1800.0,            # Max time for outer solver in seconds
        1e-6               # Gradient norm tolerance for the inner solver
    )
    println(settings_min)
end

# Example 3: Error Handling with Invalid Inputs
"""
This example shows how the validation mechanism in `SEQUOIA_Settings` works when invalid inputs are provided
for the outer method. An error is raised explaining which options are valid.

# Usage:
    example_invalid_method()

# Expected Output:
    Invalid outer method: :InvalidMethod. Valid methods are: QPM, AugLag, IntPt, SEQUOIA.
"""
function example_invalid_method()
    try
        settings_invalid = SEQUOIA_Settings(
            :InvalidMethod,     # Invalid outer method
            :LBFGS,             # Inner solver
            false,              # Feasibility
            1e-8,               # Residual tolerance
            1000,               # Max iterations
            3600.0,             # Max time
            1e-6                # Gradient norm tolerance for the inner solver
        )
    catch e
        println(e)  # Prints the error message
    end
end

# Example 4: Using Custom Solver Parameters
"""
This example demonstrates how to specify custom solver parameters when using the full constructor for `SEQUOIA_Settings`.
The `solver_params` field can accept a vector of floats to specify parameters like step sizes or penalty parameters.

# Usage:
    example_with_solver_params()

# Expected Output:
    SEQUOIA_Settings(:AugLag, :GradientDescent, false, 1.0e-6, 800, 3000.0, 1.0e-5, :MaxIterations, 100, nothing, false, nothing, nothing, 1.0e-6, [0.01, 10.0])
"""
function example_with_solver_params()
    settings_with_params = SEQUOIA_Settings(
        :AugLag,            # Outer method
        :GradientDescent,   # Inner solver
        false,              # Feasibility
        1e-6,               # Residual tolerance
        800,                # Max iterations
        3000.0,             # Max time
        1e-5,               # Gradient norm tolerance
        conv_crit = :MaxIterations, # Convergence criterion
        max_iter_inner = 100,       # Max inner iterations
        step_min = 1e-6,            # Minimum step size
        solver_params = [0.01, 10.0] # Custom solver parameters
    )
    println(settings_with_params)
end


# Call all examples
using Sequoia

example_full_constructor()
example_minimal_constructor()
example_invalid_method()
example_with_solver_params()
