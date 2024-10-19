export SolverStatus
export SEQUOIA_Solution_step
export SEQUOIA_Iterates
export add_step!
export get_all_solutions
export get_all_fvals
export get_all_convergence_metrics
export get_all_cvals
export get_all_step_sizes
export get_all_outer_iteration_numbers
export get_all_inner_iterations
export get_all_inner_comp_times
export get_all_solver_statuses

@enum SolverStatus success failed exceeded_iter exceeded_time f_and_g_not_finite stopped invalid_start not_converged infeasibility_detected positive_decrease reached_ftol reached_xtol reached_gtol

"""
    SEQUOIA_Solution_step{T}

This struct represents the state of a solution at a particular iteration in the SEQUOIA optimization solver.

# Fields

- `x::Vector{T}`: The solution vector at this iteration.
- `fval::T`: The value of the objective function at this iteration.
- `gval::Vector{T}`: The gradient of the objective function at this iteration.
- `cval::Union{Nothing, Vector{T}}`: The constraints evaluated at this iteration, or `nothing` if no constraints.
- `step_size::Union{Nothing, T}`: The step size used during this iteration, or `nothing` if not used.
- `convergence_metric::T`: A metric used to measure convergence (e.g., gradient norm).
- `outer_iteration_number::Int`: The current outer iteration number.
- `num_inner_iterations::Union{Nothing, Int}`: The number of inner iterations before exiting, or `nothing` if not applicable.
- `inner_comp_time::Float64`: The time elapsed during the inner solve call, measured in seconds.
- `solver_status::SolverStatus`: The status of the solver (e.g., `:success`, `:failed`).

# Example

```julia
step = SEQUOIA_Solution_step(
    x = [1.0, 2.0], 
    fval = 0.5, 
    gval = [0.1, 0.2], 
    cval = nothing, 
    step_size = 0.01, 
    convergence_metric = 1e-6, 
    outer_iteration_number = 10, 
    num_inner_iterations = 5, 
    inner_comp_time = 0.02, 
    solver_status = :success
)
"""
struct SEQUOIA_Solution_step{T}
    x::Vector{T}                               # Solution vector at this iteration
    fval::T                                    # Objective value at this iteration
    gval::Vector{T}                            # Gradient of the objective function
    cval::Union{Nothing, Vector{T}}            # Constraints evaluated (if any)
    step_size::Union{Nothing, T}               # Step size used in this iteration, optional
    convergence_metric::T                      # Convergence metric (e.g., gradient norm)
    outer_iteration_number::Int                # The current iteration number
    num_inner_iterations::Union{Nothing, Int}  # Number of inner iterations before exit
    inner_comp_time::Float64                   # Time elapsed in the inner solve call (in seconds)
    solver_status::SolverStatus                # Solver status as obtained from the inner solver

    # Primary Constructor
    """
    SEQUOIA_Solution_step(
        x::Vector{T},
        fval::T,
        gval::Vector{T},
        cval::Union{Nothing, Vector{T}} = nothing,
        step_size::Union{Nothing, T} = nothing,
        convergence_metric::T = zero(T),
        outer_iteration_number::Int = 0,
        num_inner_iterations::Union{Nothing, Int} = nothing,
        inner_comp_time::Float64 = 0.0,
        solver_status::SolverStatus = :success
    )

    The primary constructor for SEQUOIA_Solution_step. It takes a solution vector `x`, objective function value `fval`, and gradient `gval` as mandatory arguments. Optional fields include constraints `cval`, step size `step_size`, convergence metric, outer iteration number, number of inner iterations, inner computation time, and solver status.

    - `x`: The solution vector at this iteration.
    - `fval`: The objective function value.
    - `gval`: The gradient vector.
    - `cval`: Constraints vector (optional, defaults to `nothing`).
    - `step_size`: Step size (optional, defaults to `nothing`).
    - `convergence_metric`: Convergence metric, such as gradient norm (optional, defaults to 0.0).
    - `outer_iteration_number`: Outer iteration number (optional, defaults to 0).
    - `num_inner_iterations`: Number of inner iterations (optional, defaults to `nothing`).
    - `inner_comp_time`: Inner computation time in seconds (optional, defaults to 0.0).
    - `solver_status`: Solver status (optional, defaults to `:success`).
    """
    function SEQUOIA_Solution_step(
        x::Vector{T},
        fval::T,
        gval::Vector{T},
        cval::Union{Nothing, Vector{T}} = nothing,       # Defaults to `nothing` if not provided
        step_size::Union{Nothing, T} = nothing,          # Optional step size defaults to `nothing`
        convergence_metric::T = zero(T),                 # Defaults to zero of type T
        outer_iteration_number::Int = 0,                 # Defaults to iteration number 0 if not provided
        num_inner_iterations::Union{Nothing, Int} = nothing,  # Defaults to `nothing` if not provided
        inner_comp_time::Float64 = 0.0,                  # Defaults to 0 seconds
        solver_status::SolverStatus = :success           # Defaults to `:success` if not provided
    ) where {T}
        # Validate the input before creating the struct
        validate_arguments(x, fval, gval, cval, step_size, convergence_metric, outer_iteration_number, num_inner_iterations, inner_comp_time, solver_status)

        return new{T}(x, fval, gval, cval, step_size, convergence_metric, outer_iteration_number, num_inner_iterations, inner_comp_time, solver_status)
    end
end

# Constructor with default step size omitted
"""
SEQUOIA_Solution_step(
    x::Vector{T},
    fval::T,
    gval::Vector{T},
    convergence_metric::T = 0.0,
    outer_iteration_number::Int = 0,
    num_inner_iterations::Union{Nothing, Int} = nothing,
    inner_comp_time::Float64 = 0.0,
    solver_status::SolverStatus = :success
)

Constructor for SEQUOIA_Solution_step without step size or constraints. This version is useful when step size and constraints are not relevant for the iteration.

- `x`: The solution vector at this iteration.
- `fval`: The objective function value.
- `gval`: The gradient vector.
- `convergence_metric`: Convergence metric, such as gradient norm (optional, defaults to 0.0).
- `outer_iteration_number`: Outer iteration number (optional, defaults to 0).
- `num_inner_iterations`: Number of inner iterations (optional, defaults to `nothing`).
- `inner_comp_time`: Inner computation time in seconds (optional, defaults to 0.0).
- `solver_status`: Solver status (optional, defaults to `:success`).
"""
function SEQUOIA_Solution_step(
    x::Vector{T},
    fval::T,
    gval::Vector{T},
    convergence_metric::T = zero(T),
    outer_iteration_number::Int = 0,
    num_inner_iterations::Union{Nothing, Int} = nothing,
    inner_comp_time::Float64 = 0.0,
    solver_status::SolverStatus = :success
) where {T}
    validate_arguments(x, fval, gval, nothing, nothing, convergence_metric, outer_iteration_number, num_inner_iterations, inner_comp_time, solver_status)
    return new{T}(x, fval, gval, nothing, nothing, convergence_metric, outer_iteration_number, num_inner_iterations, inner_comp_time, solver_status)
end

# Constructor with minimal arguments (for quick instantiation)
"""
SEQUOIA_Solution_step(
    x::Vector{T},
    fval::T,
    gval::Vector{T}
)

Constructor for SEQUOIA_Solution_step with only the solution vector `x`, the objective function value `fval`, and the gradient `gval`. All other fields are set to defaults.

- `x`: The solution vector at this iteration.
- `fval`: The objective function value.
- `gval`: The gradient vector.
"""
function SEQUOIA_Solution_step(x::Vector{T}, fval::T, gval::Vector{T}) where {T}
    return SEQUOIA_Solution_step(x, fval, gval, nothing, nothing, zero(T), 0, nothing, 0.0, :success)
end

"""
    validate_arguments(x, fval, gval, cval, step_size, convergence_metric, outer_iteration_number, num_inner_iterations, inner_comp_time, solver_status)

A function that validates the input arguments for the SEQUOIA_Solution_step constructor.
"""
function validate_arguments(
    x::Vector{T},
    fval::T,
    gval::Vector{T},
    cval::Union{Nothing, Vector{T}},
    step_size::Union{Nothing, T},
    convergence_metric::T,
    outer_iteration_number::Int,
    num_inner_iterations::Union{Nothing, Int},
    inner_comp_time::Float64,
    solver_status::SolverStatus
) where {T}

    @assert !isempty(x) "Solution vector `x` cannot be empty."
    @assert !isempty(gval) "Gradient vector `gval` cannot be empty."
    @assert isnumeric(fval) "Objective function value `fval` must be numeric."
    @assert isnumeric(convergence_metric) "Convergence metric `convergence_metric` must be numeric."
    @assert inner_comp_time >= 0 "Inner computation time must be non-negative."
    @assert outer_iteration_number >= 0 "Outer iteration number must be non-negative."
    
    if !isnothing(num_inner_iterations)
        @assert num_inner_iterations >= 0 "Number of inner iterations must be non-negative."
    end
    if !isnothing(step_size)
        @assert step_size > 0 "Step size must be positive if provided."
    end
    if !isnothing(cval)
        @assert all(isnumeric, cval) "Constraint values `cval` must be a vector of numeric values or `nothing`."
    end

    valid_statuses = [success, failed, exceeded_iter, exceeded_time, f_and_g_not_finite, stopped, invalid_start, not_converged, infeasibility_detected, positive_decrease, reached_ftol, reached_xtol, reached_gtol]
    @assert solver_status in valid_statuses "Invalid solver status: $solver_status. Must be one of: $valid_statuses."
end

"""
    isnumeric(x)

A utility function to check if a value is numeric or vector of numeric values (Float64, Int, etc.).
"""
isnumeric(x) = x isa Number || (x isa AbstractArray && all(isa(x, Number) for x in x))

############################################################################
## MULTIPLE ITERATES
############################################################################

"""
    SEQUOIA_Iterates{T}

This struct stores a collection of solution steps (`SEQUOIA_Solution_step{T}`) obtained during the iterative optimization process in the SEQUOIA solver.

# Fields

- `steps::Vector{SEQUOIA_Solution_step{T}}`: A vector that holds multiple iteration steps, each represented by a `SEQUOIA_Solution_step`.

# Example

```julia
iterates = SEQUOIA_Iterates{Float64}()
step = SEQUOIA_Solution_step([1.0, 2.0], 0.5, [0.1, 0.2])
add_step!(iterates, step)
"""
struct SEQUOIA_Iterates{T}
    steps::Vector{SEQUOIA_Solution_step{T}}  # A vector to hold multiple iteration steps

    # Constructor to initialize an empty SEQUOIA_Iterates
    SEQUOIA_Iterates() = new{T}(Vector{SEQUOIA_Solution_step{T}}())
end

"""
    add_step!(iterates::SEQUOIA_Iterates{T}, step::SEQUOIA_Solution_step{T})

Add a new `SEQUOIA_Solution_step` to the `SEQUOIA_Iterates` collection. This function checks that the added step is of the correct type `T`, consistent with the type of the `SEQUOIA_Iterates` instance.

# Arguments

- `iterates::SEQUOIA_Iterates{T}`: An instance of `SEQUOIA_Iterates` holding the collection of solution steps.
- `step::SEQUOIA_Solution_step{T}`: A `SEQUOIA_Solution_step` of the same type `T` to be added to the `iterates`.

# Throws

- `AssertionError` if the type of the step does not match the type of the `iterates` collection.

# Example

```julia
iterates = SEQUOIA_Iterates{Float64}()
step = SEQUOIA_Solution_step([1.0, 2.0], 0.5, [0.1, 0.2])
add_step!(iterates, step)
"""
function add_step!(iterates::SEQUOIA_Iterates{T}, step::SEQUOIA_Solution_step{T}) where {T}
    # Ensure the step has the correct type 
    @assert isa(step, SEQUOIA_Solution_step{T}) "Step type does not match the iterates' type."
    push!(iterates.steps, step)
end

"""
    get_all_solutions(iterates::SEQUOIA_Iterates{T}) -> Vector{Vector{T}}

Return a vector of all solution vectors (`x`) from the stored steps in `SEQUOIA_Iterates`.

# Arguments

- `iterates::SEQUOIA_Iterates{T}`: An instance of `SEQUOIA_Iterates`, which holds the collection of solution steps.

# Returns

- A vector of solution vectors (`x`) from each step in the `SEQUOIA_Iterates`.

# Example

```julia
iterates = SEQUOIA_Iterates{Float64}()
step1 = SEQUOIA_Solution_step([1.0, 2.0], 0.5, [0.1, 0.2])
step2 = SEQUOIA_Solution_step([1.1, 2.1], 0.45, [0.05, 0.15])
add_step!(iterates, step1)
add_step!(iterates, step2)

all_solutions = get_all_solutions(iterates)
"""
# Functions to extract specific information
function get_all_solutions(iterates::SEQUOIA_Iterates{T}) where {T}
    [step.x for step in iterates.steps]
end

"""
    get_all_fvals(iterates::SEQUOIA_Iterates{T}) -> Vector{T}

Return a vector of all objective function values (`fval`) from the stored steps in `SEQUOIA_Iterates`.

# Arguments

- `iterates::SEQUOIA_Iterates{T}`: An instance of `SEQUOIA_Iterates` that holds the collection of solution steps. Each step contains an objective function value (`fval`).

# Returns

- A `Vector{T}` where each element is the objective function value (`fval`) from each step in the `SEQUOIA_Iterates`.

# Example

```julia
iterates = SEQUOIA_Iterates{Float64}()
step1 = SEQUOIA_Solution_step([1.0, 2.0], 0.5, [0.1, 0.2])
step2 = SEQUOIA_Solution_step([1.1, 2.1], 0.45, [0.05, 0.15])
add_step!(iterates, step1)
add_step!(iterates, step2)

all_fvals = get_all_fvals(iterates)
"""

function get_all_fvals(iterates::SEQUOIA_Iterates{T}) where {T}
    [step.fval for step in iterates.steps]
end

"""
    get_all_convergence_metrics(iterates::SEQUOIA_Iterates{T}) -> Vector{T}

Return a vector of all convergence metrics (`convergence_metric`) from the stored steps in `SEQUOIA_Iterates`.

# Arguments

- `iterates::SEQUOIA_Iterates{T}`: An instance of `SEQUOIA_Iterates` that holds the collection of solution steps. Each step contains a convergence metric.

# Returns

- A `Vector{T}` where each element is the convergence metric (`convergence_metric`) from each step in the `SEQUOIA_Iterates`.

# Example

```julia
iterates = SEQUOIA_Iterates{Float64}()
step1 = SEQUOIA_Solution_step([1.0, 2.0], 0.5, [0.1, 0.2], convergence_metric=1e-6)
step2 = SEQUOIA_Solution_step([1.1, 2.1], 0.45, [0.05, 0.15], convergence_metric=1e-5)
add_step!(iterates, step1)
add_step!(iterates, step2)

all_metrics = get_all_convergence_metrics(iterates)
"""

function get_all_convergence_metrics(iterates::SEQUOIA_Iterates{T}) where {T}
    [step.convergence_metric for step in iterates.steps]
end

"""
    get_all_cvals(iterates::SEQUOIA_Iterates{T}) -> Vector{Union{Nothing, Vector{T}}}

Return a vector of all constraint evaluations (`cval`) from the stored steps in `SEQUOIA_Iterates`.

# Arguments

- `iterates::SEQUOIA_Iterates{T}`: An instance of `SEQUOIA_Iterates` that holds the collection of solution steps. Each step contains a constraint evaluation (`cval`), which can either be a vector of values or `nothing` if no constraints were evaluated.

# Returns

- A `Vector{Union{Nothing, Vector{T}}}` where each element is either a vector of constraint evaluations or `nothing` from each step in the `SEQUOIA_Iterates`.

# Example

```julia
iterates = SEQUOIA_Iterates{Float64}()
step1 = SEQUOIA_Solution_step([1.0, 2.0], 0.5, [0.1, 0.2], cval=[0.05, 0.1])
step2 = SEQUOIA_Solution_step([1.1, 2.1], 0.45, [0.05, 0.15], cval=nothing)
add_step!(iterates, step1)
add_step!(iterates, step2)

all_cvals = get_all_cvals(iterates)
"""
# Get all constraint evaluations (cval) from the stored steps
function get_all_cvals(iterates::SEQUOIA_Iterates{T}) where {T}
    [step.cval for step in iterates.steps]
end


"""
    get_all_step_sizes(iterates::SEQUOIA_Iterates{T}) -> Vector{Union{Nothing, T}}

Return a vector of all step sizes (`step_size`) from the stored steps in `SEQUOIA_Iterates`.

# Arguments

- `iterates::SEQUOIA_Iterates{T}`: An instance of `SEQUOIA_Iterates` that holds the collection of solution steps. Each step may contain a step size (`step_size`), which can either be a numeric value of type `T` or `nothing` if no step size is used.

# Returns

- A `Vector{Union{Nothing, T}}` where each element is either the step size or `nothing` from each step in the `SEQUOIA_Iterates`.

# Example

```julia
iterates = SEQUOIA_Iterates{Float64}()
step1 = SEQUOIA_Solution_step([1.0, 2.0], 0.5, [0.1, 0.2], step_size=0.01)
step2 = SEQUOIA_Solution_step([1.1, 2.1], 0.45, [0.05, 0.15], step_size=nothing)
add_step!(iterates, step1)
add_step!(iterates, step2)

all_step_sizes = get_all_step_sizes(iterates)

"""

# Get all step sizes from the stored steps
function get_all_step_sizes(iterates::SEQUOIA_Iterates{T}) where {T}
    [step.step_size for step in iterates.steps]
end

"""
    get_all_outer_iteration_numbers(iterates::SEQUOIA_Iterates{T}) -> Vector{Int}

Return a vector of all outer iteration numbers (`outer_iteration_number`) from the stored steps in `SEQUOIA_Iterates`.

# Arguments

- `iterates::SEQUOIA_Iterates{T}`: An instance of `SEQUOIA_Iterates` that holds the collection of solution steps. Each step contains an outer iteration number (`outer_iteration_number`), which tracks the current iteration number in the optimization process.

# Returns

- A `Vector{Int}` where each element is the outer iteration number from each step in the `SEQUOIA_Iterates`.

# Example

```julia
iterates = SEQUOIA_Iterates{Float64}()
step1 = SEQUOIA_Solution_step([1.0, 2.0], 0.5, [0.1, 0.2], outer_iteration_number=1)
step2 = SEQUOIA_Solution_step([1.1, 2.1], 0.45, [0.05, 0.15], outer_iteration_number=2)
add_step!(iterates, step1)
add_step!(iterates, step2)

all_outer_iterations = get_all_outer_iteration_numbers(iterates)
"""

# Get all outer iteration numbers from the stored steps
function get_all_outer_iteration_numbers(iterates::SEQUOIA_Iterates{T}) where {T}
    [step.outer_iteration_number for step in iterates.steps]
end

"""
    get_all_inner_iterations(iterates::SEQUOIA_Iterates{T}) -> Vector{Union{Nothing, Int}}

Return a vector of all numbers of inner iterations (`num_inner_iterations`) from the stored steps in `SEQUOIA_Iterates`.

# Arguments

- `iterates::SEQUOIA_Iterates{T}`: An instance of `SEQUOIA_Iterates` that holds the collection of solution steps. Each step may contain a number of inner iterations (`num_inner_iterations`), which can either be an integer or `nothing` if no inner iterations were recorded for that step.

# Returns

- A `Vector{Union{Nothing, Int}}` where each element is either the number of inner iterations (`Int`) or `nothing` from each step in the `SEQUOIA_Iterates`.

# Example

```julia
iterates = SEQUOIA_Iterates{Float64}()
step1 = SEQUOIA_Solution_step([1.0, 2.0], 0.5, [0.1, 0.2], num_inner_iterations=5)
step2 = SEQUOIA_Solution_step([1.1, 2.1], 0.45, [0.05, 0.15], num_inner_iterations=nothing)
add_step!(iterates, step1)
add_step!(iterates, step2)

all_inner_iterations = get_all_inner_iterations(iterates)

"""

# Get all numbers of inner iterations from the stored steps
function get_all_inner_iterations(iterates::SEQUOIA_Iterates{T}) where {T}
    [step.num_inner_iterations for step in iterates.steps]
end

"""
    get_all_inner_comp_times(iterates::SEQUOIA_Iterates{T}) -> Vector{Float64}

Return a vector of all inner computation times (`inner_comp_time`) from the stored steps in `SEQUOIA_Iterates`.

# Arguments

- `iterates::SEQUOIA_Iterates{T}`: An instance of `SEQUOIA_Iterates` that holds the collection of solution steps. Each step contains an `inner_comp_time` representing the time taken for the inner solve operation in seconds.

# Returns

- A `Vector{Float64}` where each element is the inner computation time (`inner_comp_time`) from each step in the `SEQUOIA_Iterates`.

# Example

```julia
iterates = SEQUOIA_Iterates{Float64}()
step1 = SEQUOIA_Solution_step([1.0, 2.0], 0.5, [0.1, 0.2], inner_comp_time=0.02)
step2 = SEQUOIA_Solution_step([1.1, 2.1], 0.45, [0.05, 0.15], inner_comp_time=0.03)
add_step!(iterates, step1)
add_step!(iterates, step2)

all_comp_times = get_all_inner_comp_times(iterates)

"""

# Get all inner computation times from the stored steps
function get_all_inner_comp_times(iterates::SEQUOIA_Iterates{T}) where {T}
    [step.inner_comp_time for step in iterates.steps]
end

"""
    get_all_solver_statuses(iterates::SEQUOIA_Iterates{T}) -> Vector{SolverStatus}

Return a vector of all solver statuses (`solver_status`) from the stored steps in `SEQUOIA_Iterates`.

# Arguments

- `iterates::SEQUOIA_Iterates{T}`: An instance of `SEQUOIA_Iterates` that holds the collection of solution steps. Each step contains a `solver_status` indicating the status of the solver (e.g., `:success`, `:failed`, `:exceeded_iter`, etc.).

# Returns

- A `Vector{SolverStatus}` where each element is the solver status (`solver_status`) from each step in the `SEQUOIA_Iterates`.

# Example

```julia
iterates = SEQUOIA_Iterates{Float64}()
step1 = SEQUOIA_Solution_step([1.0, 2.0], 0.5, [0.1, 0.2], solver_status=:success)
step2 = SEQUOIA_Solution_step([1.1, 2.1], 0.45, [0.05, 0.15], solver_status=:failed)
add_step!(iterates, step1)
add_step!(iterates, step2)

all_statuses = get_all_solver_statuses(iterates)

"""

# Get all solver statuses from the stored steps
function get_all_solver_statuses(iterates::SEQUOIA_Iterates{T}) where {T}
    [step.solver_status for step in iterates.steps]
end

