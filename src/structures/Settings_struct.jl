export SEQUOIA_Settings

export LBFGS, BFGS, Newton, GradientDescent, NelderMead
export SEQUOIA, QPM, AugLag, IntPt

# Custom types for allowed inner solvers and outer methods
abstract type InnerSolver end
struct LBFGS <: InnerSolver end
struct BFGS <: InnerSolver end
struct Newton <: InnerSolver end
struct GradientDescent <: InnerSolver end
struct NelderMead <: InnerSolver end

abstract type OuterMethod end
struct SEQUOIA <: OuterMethod end
struct QPM <: OuterMethod end
struct AugLag <: OuterMethod end
struct IntPt <: OuterMethod end

"""
    SEQUOIA_Settings

The `SEQUOIA_Settings` struct stores the main settings required for solving optimization problems with the SEQUOIA method. These settings control the solver algorithms, convergence criteria, and various numerical tolerances. 

# Fields
- `inner_solver::InnerSolver`: Specifies the algorithm used to solve unconstrained subproblems. Valid options include:
    - `LBFGS`: Limited-memory Broyden–Fletcher–Goldfarb–Shanno algorithm.
    - `BFGS`: Broyden–Fletcher–Goldfarb–Shanno algorithm.
    - `Newton`: Newton's method.
    - `GradientDescent`: Gradient descent method.
    - `NelderMead`: Nelder-Mead simplex method.

- `max_iter::Int`: Maximum number of iterations allowed for the outer solver. It must be a positive integer.

- `max_time::Real`: Maximum computational time (in seconds). The time must be non-negative or `Inf` for no limit.

- `resid_tolerance::Real`: Residual tolerance for constraints. The solver will consider the constraints satisfied when the residual is below this value. Must be a positive number.

- `cost_tolerance::Real`: Desired optimality gap for convergence. The solver will stop when the difference between the current solution and the optimal cost is below this threshold. Must be a positive number.

- `cost_min::Real`: Minimum allowed cost value. This is useful for detecting potentially unbounded problems. While negative values are allowed, they should not be too extreme.

- `outer_method::OuterMethod`: Specifies the outer optimization method. Valid options include:
    - `SEQUOIA`: Sequential quadratic programming-based method.
    - `QPM`: Quadratic programming method.
    - `AugLag`: Augmented Lagrangian method.
    - `IntPt`: Interior point method.

- `feasibility::Bool`: Determines if the solver is solving a feasibility problem (i.e., only ensuring constraints are satisfied) or optimizing an objective function.

- `step_size::Union{Nothing, Float64}`: Optional step size for the optimization algorithm. If not provided, the solver may use an adaptive step size.

# Example

```julia
settings = SEQUOIA_Settings(inner_solver=BFGS(), max_iter=2000, max_time=60.0, 
                            resid_tolerance=1e-8, cost_tolerance=1e-4, cost_min=-1e6, 
                            outer_method=QPM(), feasibility=false, step_size=0.1)
"""

mutable struct SEQUOIA_Settings
    inner_solver::InnerSolver           # The inner solver used for unconstrained problems.
    max_iter::Int                       # Maximum number of outer iterations allowed.
    max_time::Real                      # Maximum computational time (in seconds).

    resid_tolerance::Real               # Residual tolerance (i.e., when constraints are considered satisfied).
    cost_tolerance::Real                # Desired optimality gap.
    cost_min::Real                      # Minimum cost - useful for spotting possible unbounded problems.

    outer_method::OuterMethod           # Optimization method (SEQUOIA, QPM, AugLag, IntPt).
    feasibility::Bool                   # Solve feasibility problem (true) or account for an objective (false)?

    step_size::Union{Nothing, Float64}  # Optional field for step size, can be nothing.

    """
    SEQUOIA_Settings(; inner_solver=LBFGS(), max_iter=3000, max_time=Inf, 
                      resid_tolerance=1e-6, cost_tolerance=1e-2, 
                      cost_min=-1e10, outer_method=SEQUOIA(), feasibility=true, 
                      step_size=nothing)

    Construct a `SEQUOIA_Settings` object with full control over all fields. 
    If not specified, default values will be used.

    # Arguments
    - `inner_solver`: Algorithm for solving the unconstrained problem. Default is `LBFGS()`.
    - `max_iter`: Maximum number of iterations. Default is 3000.
    - `max_time`: Maximum allowed computational time (seconds). Default is `Inf`.
    - `resid_tolerance`: Residual tolerance for constraint satisfaction. Default is `1e-6`.
    - `cost_tolerance`: Desired optimality gap. Default is `1e-2`.
    - `cost_min`: Minimum cost value (used for detecting unbounded problems). Default is `-1e10`.
    - `outer_method`: Outer optimization method (e.g., `SEQUOIA`, `QPM`, `AugLag`, `IntPt`). Default is `SEQUOIA()`.
    - `feasibility`: Whether the problem is purely a feasibility problem. Default is `true`.
    - `step_size`: Optional step size for optimization. If not provided, the solver may use an adaptive step size.
    """

    # Constructor with keyword arguments and default values
    function SEQUOIA_Settings(;inner_solver::InnerSolver=LBFGS(),
                               max_iter::Int=3000,
                               max_time::Real=Inf,
                               resid_tolerance::Real=1e-6,
                               cost_tolerance::Real=1e-2,
                               cost_min::Real=-1e10,
                               outer_method::OuterMethod=SEQUOIA(),
                               feasibility::Bool=true,
                               step_size::Union{Nothing, Float64}=nothing)
        # Call validation functions
        validate_inner_solver(inner_solver)
        validate_outer_method(outer_method)
        validate_max_iter(max_iter)
        validate_max_time(max_time)
        validate_tolerance(resid_tolerance, "resid_tolerance")
        validate_tolerance(cost_tolerance, "cost_tolerance")
        validate_cost_min(cost_min)
        validate_step_size(step_size)

        # Initialize the struct if all validations pass
        return new(inner_solver, max_iter, max_time, resid_tolerance, cost_tolerance, cost_min, outer_method, feasibility, step_size)
    end

    """
    SEQUOIA_Settings(inner_solver::InnerSolver, outer_method::OuterMethod; max_iter=1000, max_time=Inf)

    Construct a `SEQUOIA_Settings` object with the specified `inner_solver` and `outer_method`, leaving other fields to their default values.

    # Arguments
    - `inner_solver`: Algorithm for solving the unconstrained problem.
    - `outer_method`: Outer optimization method.
    - `max_iter`: Maximum number of iterations. Default is 1000.
    - `max_time`: Maximum computational time (seconds). Default is `Inf`.
    """

    # Simplified constructor with just the most important fields
    function SEQUOIA_Settings(inner_solver::InnerSolver, outer_method::OuterMethod; max_iter::Int=1000, max_time::Real=Inf)
        SEQUOIA_Settings(inner_solver=inner_solver, max_iter=max_iter, max_time=max_time, outer_method=outer_method)
    end

    """
        SEQUOIA_Settings(inner_solver::InnerSolver; max_iter=1000, max_time=Inf)

    Construct a `SEQUOIA_Settings` object with the specified `inner_solver` and leave the outer method as the default `SEQUOIA()`.

    # Arguments
    - `inner_solver`: Algorithm for solving the unconstrained problem.
    - `max_iter`: Maximum number of iterations. Default is 1000.
    - `max_time`: Maximum computational time (seconds). Default is `Inf`.
    """

    # Constructor with only inner_solver and default method
    function SEQUOIA_Settings(inner_solver::InnerSolver; max_iter::Int=1000, max_time::Real=Inf)
        SEQUOIA_Settings(inner_solver=inner_solver, max_iter=max_iter, max_time=max_time, outer_method=SEQUOIA())
    end
end




# Validation for inner_solver
function validate_inner_solver(inner_solver::InnerSolver)
    valid_solvers = [LBFGS, BFGS, Newton, GradientDescent, NelderMead]
    if !(typeof(inner_solver) in valid_solvers)
        throw(ArgumentError("Invalid inner solver: $(typeof(inner_solver)). Valid solvers are: $(valid_solvers)."))
    end
end

# Validation for outer_method
function validate_outer_method(outer_method::OuterMethod)
    valid_methods = [SEQUOIA, QPM, AugLag, IntPt]
    if !(typeof(outer_method) in valid_methods)
        throw(ArgumentError("Invalid outer method: $(typeof(outer_method)). Valid methods are: $(valid_methods)."))
    end
end

# Validation for max_iter
function validate_max_iter(max_iter::Int)
    if max_iter <= 0
        throw(ArgumentError("max_iter must be a positive integer. Given: $max_iter"))
    end
end

# Validation for max_time
function validate_max_time(max_time::Real)
    if max_time < 0
        throw(ArgumentError("max_time must be non-negative or Inf. Given: $max_time"))
    end
end

# Generic validation for tolerance (residual and cost tolerance)
function validate_tolerance(tolerance::Real, name::String)
    if tolerance <= 0
        throw(ArgumentError("$name must be a positive number. Given: $tolerance"))
    end
end

# Validation for cost_min
function validate_cost_min(cost_min::Real)
    # This is somewhat arbitrary but can help catch unreasonable values
    if cost_min < -1e15
        throw(ArgumentError("cost_min is too low, suggesting potential issues. Given: $cost_min"))
    end
end

# Validation for step_size
function validate_step_size(step_size::Union{Nothing, Float64})
    if step_size !== nothing && step_size <= 0
        throw(ArgumentError("step_size must be a positive number or nothing. Given: $step_size"))
    end
end
