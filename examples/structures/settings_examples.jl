"""
# Example 1: Using the Full Constructor

This example shows how to use the full constructor for `SEQUOIA_Settings`, where all fields are specified, including inner and outer solvers, convergence criteria, tolerances, and optional parameters.

```julia
example_full_constructor()

settings_full = SEQUOIA_Settings(
    :SEQUOIA,          # Outer method
    :LBFGS,            # Inner solver
    false,             # Feasibility: solving an optimization problem, not just feasibility
    1e-8,              # Residual tolerance for constraints
    1000,              # Max iterations for outer solver
    3600.0,            # Max time for outer solver in seconds
    :GradientNorm,     # Convergence criterion: based on gradient norm
    nothing,           # Max iterations for inner solver (nothing means default)
    nothing,           # Max time for inner solver (nothing means no time limit)
    1e-4,              # Cost tolerance: the solver will stop if cost difference is below this
    -1e6,              # Minimum cost to help detect unbounded problems
    [1.0, 0.5]         # Optional solver parameters (e.g., penalty parameters, step sizes)
)

println(settings_full)

Example output: 

SEQUOIA_Settings(:SEQUOIA, :LBFGS, false, 1.0e-8, 1000, 3600.0, :GradientNorm, nothing, nothing, 1.0e-4, -1000000.0, [1.0, 0.5])

"""
function example_full_constructor()
    settings_full = SEQUOIA_Settings(
        :SEQUOIA,          # Outer method
        :LBFGS,            # Inner solver
        false,             # Feasibility: solving an optimization problem, not just feasibility
        1e-8,              # Residual tolerance for constraints
        1000,              # Max iterations for outer solver
        3600.0,            # Max time for outer solver in seconds
        :GradientNorm,     # Convergence criterion: based on gradient norm
        nothing,           # Max iterations for inner solver (nothing means default)
        nothing,           # Max time for inner solver (nothing means no time limit)
        1e-4,              # Cost tolerance: the solver will stop if cost difference is below this
        -1e6,              # Minimum cost to help detect unbounded problems
        [1.0, 0.5]         # Optional solver parameters (e.g., penalty parameters, step sizes)
    )
    println(settings_full)
end

"""
# Example 2: Using the Minimal Constructor

This example demonstrates the minimal constructor for `SEQUOIA_Settings`, which defaults some fields such as convergence criteria (`:GradientNorm`), and omits optional parameters like `cost_tolerance` and `solver_params`.

```julia
example_minimal_constructor()

settings_min = SEQUOIA_Settings(
    :QPM,              # Outer method
    :Newton,           # Inner solver
    true,              # Feasibility: solving a feasibility problem
    1e-6,              # Residual tolerance for constraints
    500,               # Max iterations for outer solver
    1800.0             # Max time for outer solver in seconds
)

println(settings_min)

Example output:

SEQUOIA_Settings(:QPM, :Newton, true, 1.0e-6, 500, 1800.0, :GradientNorm, nothing, nothing, nothing, nothing, nothing)

"""
function example_minimal_constructor()
    settings_min = SEQUOIA_Settings(
        :QPM,              # Outer method
        :Newton,           # Inner solver
        true,              # Feasibility: solving a feasibility problem
        1e-6,              # Residual tolerance for constraints
        500,               # Max iterations for outer solver
        1800.0             # Max time for outer solver in seconds
    )
    println(settings_min)
end

"""
# Example 3: Error Handling with Invalid Inputs

This example shows how the validation mechanism in `SEQUOIA_Settings` works when invalid inputs are provided for the outer method. An error is raised explaining which options are valid.

```julia
example_invalid_method()

settings_min = SEQUOIA_Settings(
    :QPM,              # Outer method
    :Newton,           # Inner solver
    true,              # Feasibility: solving a feasibility problem
    1e-6,              # Residual tolerance for constraints
    500,               # Max iterations for outer solver
    1800.0             # Max time for outer solver in seconds
)

println(settings_min)

Example output:

ErrorException("Invalid outer method: InvalidMethod. Valid options are: SEQUOIA, QPM, AugLag, IntPt")

"""
function example_minimal_constructor()
    settings_min = SEQUOIA_Settings(
        :QPM,              # Outer method
        :Newton,           # Inner solver
        true,              # Feasibility: solving a feasibility problem
        1e-6,              # Residual tolerance for constraints
        500,               # Max iterations for outer solver
        1800.0             # Max time for outer solver in seconds
    )
    println(settings_min)
end

"""
# Example 3: Error Handling with Invalid Inputs

This example shows how the validation mechanism in `SEQUOIA_Settings` works when invalid inputs are provided for the outer method. An error is raised explaining which options are valid.

```julia
example_invalid_method()

try
    settings_invalid = SEQUOIA_Settings(
        :InvalidMethod,     # Invalid outer method (will raise an error)
        :LBFGS,             # Inner solver
        false,              # Feasibility
        1e-8,               # Residual tolerance
        1000,               # Max iterations
        3600.0              # Max time
    )
catch e
    println(e)  # Prints the error message: Invalid outer method: :InvalidMethod
end

Example output:

Invalid outer method: :InvalidMethod. Valid options are: SEQUOIA, QPM, AugLag, IntPt

"""
function example_invalid_method()
    try
        settings_invalid = SEQUOIA_Settings(
            :InvalidMethod,     # Invalid outer method (will raise an error)
            :LBFGS,             # Inner solver
            false,              # Feasibility
            1e-8,               # Residual tolerance
            1000,               # Max iterations
            3600.0              # Max time
        )
    catch e
        println(e)  # Prints the error message: Invalid outer method: :InvalidMethod
    end
end

"""
# Example 4: Using Custom Solver Parameters

This example demonstrates how to specify custom solver parameters when using the full constructor for `SEQUOIA_Settings`. The `solver_params` field can accept a vector of floats to specify parameters like step sizes or penalty parameters.

```julia
example_with_solver_params()

settings_with_params = SEQUOIA_Settings(
    :AugLag,            # Outer method
    :GradientDescent,   # Inner solver
    false,              # Feasibility
    1e-6,               # Residual tolerance
    800,                # Max iterations
    3000.0,             # Max time
    :MaxIterations,     # Convergence based on max iterations
    100,                # Max inner iterations
    nothing,            # No time limit for inner solver
    1e-5,               # Cost tolerance
    nothing,            # No minimum cost
    [0.01, 10.0]        # Custom solver parameters (step size and penalty parameter)
)

println(settings_with_params)

Example output:

SEQUOIA_Settings(:AugLag, :GradientDescent, false, 1.0e-6, 800, 3000.0, :MaxIterations, 100, nothing, 1.0e-5, nothing, [0.01, 10.0])

"""

function example_with_solver_params()
    settings_with_params = SEQUOIA_Settings(
        :AugLag,            # Outer method
        :GradientDescent,   # Inner solver
        false,              # Feasibility
        1e-6,               # Residual tolerance
        800,                # Max iterations
        3000.0,             # Max time
        :MaxIterations,     # Convergence based on max iterations
        100,                # Max inner iterations
        nothing,            # No time limit for inner solver
        1e-5,               # Cost tolerance
        nothing,            # No minimum cost
        [0.01, 10.0]        # Custom solver parameters (step size and penalty parameter)
    )
    println(settings_with_params)
end
