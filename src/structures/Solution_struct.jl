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

# Define solver statuses as a list of allowed symbols
SolverStatus = [
    :success,
    :failed,
    :exceeded_iter,
    :exceeded_time,
    :f_and_g_not_finite,
    :stopped,
    :invalid_start,
    :not_converged,
    :infeasibility_detected,
    :positive_decrease,
    :reached_ftol,
    :reached_xtol,
    :reached_gtol
]

"""
    SEQUOIA_Solution_step

This struct represents the state of a solution at a particular iteration in the SEQUOIA optimization solver.

# Fields

- `outer_iteration_number::Int`: The current outer iteration number.
- `convergence_metric::Float64`: A metric used to measure convergence (e.g., feasibility for QPM, gradient norm for AugLag, or optimality gap in SEQUOIA).
- `solver_status::Symbol`: The status of the solver (e.g., `:success`, `:failed`).
- `inner_comp_time::Float64`: The time elapsed during the inner solve call, measured in seconds.
- `num_inner_iterations::Int`: The number of inner iterations before exiting.
- `x::Vector{Float64}`: The solution vector at this iteration.
- `fval::Float64`: The value of the objective function at this iteration.
- `gval::Vector{Float64}`: The gradient of the objective function at this iteration.
- `cval::Union{Nothing, Vector{Float64}}`: The constraints evaluated at this iteration, or `nothing` if no constraints.
- `solver_params::Union{Nothing, Vector{Float64}}`: Optional solver-related parameters.
- `x_iterates::Union{Nothing, Vector{Vector{Float64}}}`: A vector storing the history of all inner `x` iterates during this step.
"""
struct SEQUOIA_Solution_step
    outer_iteration_number::Int                                     # The current outer iteration number
    convergence_metric::Float64                                     # Convergence metric (e.g., feasibility, gradient norm, or optimality gap)
    solver_status::Symbol                                           # Solver status as obtained from the inner solver
    inner_comp_time::Float64                                        # Time elapsed in the inner solve call (in seconds)
    num_inner_iterations::Int                                       # Number of inner iterations before exit
    x::Vector{Float64}                                              # Solution vector at this iteration
    fval::Float64                                                   # Objective value at this iteration
    gval::Vector{Float64}                                           # Gradient of the objective function
    cval::Union{Nothing, Vector{Float64}}                           # Constraints evaluated (if any)
    solver_params::Union{Nothing, Vector{Float64}}                  # Optional solver-related parameters
    x_iterates::Union{Nothing, Vector{Vector{Float64}}}             # History of x values

    # Full constructor
    """
    SEQUOIA_Solution_step(
        outer_iteration_number::Int,
        convergence_metric::Float64,
        solver_status::Symbol,
        inner_comp_time::Float64,
        num_inner_iterations::Int,
        x::Vector{Float64},
        fval::Float64,
        gval::Vector{Float64},
        cval::Union{Nothing, Vector{Float64}} = nothing,
        solver_params::Union{Nothing, Vector{Float64}} = nothing,
        x_iterates::Union{Nothing, Vector{Vector{Float64}}} = nothing
    )

    The full constructor for SEQUOIA_Solution_step. It takes all fields, including optional ones, and validates the solution using `validate_sequoia_solution!`.

    - `outer_iteration_number`: The current outer iteration number.
    - `convergence_metric`: A metric used to measure convergence.
    - `solver_status`: Solver status (optional, defaults to `:success`).
    - `inner_comp_time`: Inner computation time in seconds.
    - `num_inner_iterations`: The number of inner iterations before exit.
    - `x`: The solution vector at this iteration.
    - `fval`: The objective function value.
    - `gval`: The gradient vector.
    - `cval`: Constraints vector (optional, defaults to `nothing`).
    - `solver_params`: Optional solver-related parameters (defaults to `nothing`).
    - `x_iterates`: A vector to store the history of `x` values at this step (defaults to `nothing`).
    """
    function SEQUOIA_Solution_step(
        outer_iteration_number::Int,
        convergence_metric::Float64,
        solver_status::Symbol,
        inner_comp_time::Float64,
        num_inner_iterations::Int,
        x::Vector{Float64},
        fval::Float64,
        gval::Vector{Float64},
        cval::Union{Nothing, Vector{Float64}} = nothing,
        solver_params::Union{Nothing, Vector{Float64}} = nothing,
        x_iterates::Union{Nothing, Vector{Vector{Float64}}} = nothing
    )
        # Create the solution instance
        solution = new(outer_iteration_number, convergence_metric, solver_status, inner_comp_time, num_inner_iterations, x, fval, gval, cval, solver_params, x_iterates)
        
        # Validate the solution
        validate_sequoia_solution!(solution) #For testing only, comment-out when releasing

        # Return the valid solution
        return solution
    end
end

############################################################################
## MULTIPLE ITERATES
############################################################################

"""
    SEQUOIA_Iterates

This struct stores a collection of solution steps (`SEQUOIA_Solution_step`) obtained during the iterative optimization process in the SEQUOIA solver.

# Fields

- `steps::Vector{SEQUOIA_Solution_step}`: A vector that holds multiple iteration steps, each represented by a `SEQUOIA_Solution_step`.

# Example

```julia
iterates = SEQUOIA_Iterates()
step = SEQUOIA_Solution_step([1.0, 2.0], 0.5, [0.1, 0.2])
add_step!(iterates, step)
"""
struct SEQUOIA_Iterates
    steps::Vector{SEQUOIA_Solution_step}  # A vector to hold multiple iteration steps

    # Constructor to initialize an empty SEQUOIA_Iterates
    SEQUOIA_Iterates() = new(Vector{SEQUOIA_Solution_step}())
end

"""
    add_step!(iterates::SEQUOIA_Iterates, step::SEQUOIA_Solution_step)

Add a new `SEQUOIA_Solution_step` to the `SEQUOIA_Iterates` collection. This function checks that the added step is of the correct type, consistent with the type of the `SEQUOIA_Iterates` instance.

# Arguments

- `iterates::SEQUOIA_Iterates`: An instance of `SEQUOIA_Iterates` holding the collection of solution steps.
- `step::SEQUOIA_Solution_step`: A `SEQUOIA_Solution_step` to be added to the `iterates`.

# Throws

- `AssertionError` if the step is not of the correct type.

# Example

```julia
iterates = SEQUOIA_Iterates()
step = SEQUOIA_Solution_step([1.0, 2.0], 0.5, [0.1, 0.2])
add_step!(iterates, step)
"""
function add_step!(iterates::SEQUOIA_Iterates, step::SEQUOIA_Solution_step)
    push!(iterates.steps, step)
end

# Functions to extract specific information
"""
    get_all_solutions(iterates::SEQUOIA_Iterates) -> Vector{Vector{Float64}}

Return a vector of all solution vectors (`x`) from the stored steps in `SEQUOIA_Iterates`.

# Example

```julia
iterates = SEQUOIA_Iterates()
step1 = SEQUOIA_Solution_step([1.0, 2.0], 0.5, [0.1, 0.2])
step2 = SEQUOIA_Solution_step([1.1, 2.1], 0.45, [0.05, 0.15])
add_step!(iterates, step1)
add_step!(iterates, step2)

all_solutions = get_all_solutions(iterates)
"""
function get_all_solutions(iterates::SEQUOIA_Iterates)
    [step.x for step in iterates.steps]
end

"""
    get_all_fvals(iterates::SEQUOIA_Iterates) -> Vector{Float64}

Return a vector of all objective function values (`fval`) from the stored steps in `SEQUOIA_Iterates`.

# Example

```julia
iterates = SEQUOIA_Iterates()
step1 = SEQUOIA_Solution_step([1.0, 2.0], 0.5, [0.1, 0.2])
step2 = SEQUOIA_Solution_step([1.1, 2.1], 0.45, [0.05, 0.15])
add_step!(iterates, step1)
add_step!(iterates, step2)

all_fvals = get_all_fvals(iterates)
"""
function get_all_fvals(iterates::SEQUOIA_Iterates)
    [step.fval for step in iterates.steps]
end

"""
    get_all_convergence_metrics(iterates::SEQUOIA_Iterates) -> Vector{Float64}

Return a vector of all convergence metrics (`convergence_metric`) from the stored steps in `SEQUOIA_Iterates`.

# Example

```julia
iterates = SEQUOIA_Iterates()
step1 = SEQUOIA_Solution_step([1.0, 2.0], 0.5, [0.1, 0.2], convergence_metric=1e-6)
step2 = SEQUOIA_Solution_step([1.1, 2.1], 0.45, [0.05, 0.15], convergence_metric=1e-5)
add_step!(iterates, step1)
add_step!(iterates, step2)

all_metrics = get_all_convergence_metrics(iterates)
"""
function get_all_convergence_metrics(iterates::SEQUOIA_Iterates)
    [step.convergence_metric for step in iterates.steps]
end

"""
    get_all_cvals(iterates::SEQUOIA_Iterates) -> Vector{Union{Nothing, Vector{Float64}}}

Return a vector of all constraint evaluations (`cval`) from the stored steps in `SEQUOIA_Iterates`.

# Example

```julia
iterates = SEQUOIA_Iterates()
step1 = SEQUOIA_Solution_step([1.0, 2.0], 0.5, [0.1, 0.2], cval=[0.05, 0.1])
step2 = SEQUOIA_Solution_step([1.1, 2.1], 0.45, [0.05, 0.15], cval=nothing)
add_step!(iterates, step1)
add_step!(iterates, step2)

all_cvals = get_all_cvals(iterates)
"""
function get_all_cvals(iterates::SEQUOIA_Iterates)
    [step.cval for step in iterates.steps]
end

"""
    get_all_step_sizes(iterates::SEQUOIA_Iterates) -> Vector{Union{Nothing, Float64}}

Return a vector of all step sizes (`step_size`) from the stored steps in `SEQUOIA_Iterates`.

# Example

```julia
iterates = SEQUOIA_Iterates()
step1 = SEQUOIA_Solution_step([1.0, 2.0], 0.5, [0.1, 0.2], step_size=0.01)
step2 = SEQUOIA_Solution_step([1.1, 2.1], 0.45, [0.05, 0.15], step_size=nothing)
add_step!(iterates, step1)
add_step!(iterates, step2)

all_step_sizes = get_all_step_sizes(iterates)
"""
function get_all_step_sizes(iterates::SEQUOIA_Iterates)
    [step.step_size for step in iterates.steps]
end

"""
    get_all_outer_iteration_numbers(iterates::SEQUOIA_Iterates) -> Vector{Int}

Return a vector of all outer iteration numbers (`outer_iteration_number`) from the stored steps in `SEQUOIA_Iterates`.

# Example

```julia
iterates = SEQUOIA_Iterates()
step1 = SEQUOIA_Solution_step([1.0, 2.0], 0.5, [0.1, 0.2], outer_iteration_number=1)
step2 = SEQUOIA_Solution_step([1.1, 2.1], 0.45, [0.05, 0.15], outer_iteration_number=2)
add_step!(iterates, step1)
add_step!(iterates, step2)

all_outer_iterations = get_all_outer_iteration_numbers(iterates)
"""
function get_all_outer_iteration_numbers(iterates::SEQUOIA_Iterates)
    [step.outer_iteration_number for step in iterates.steps]
end

"""
    get_all_inner_iterations(iterates::SEQUOIA_Iterates) -> Vector{Union{Nothing, Int}}

Return a vector of all numbers of inner iterations (`num_inner_iterations`) from the stored steps in `SEQUOIA_Iterates`.

# Example

```julia
iterates = SEQUOIA_Iterates()
step1 = SEQUOIA_Solution_step([1.0, 2.0], 0.5, [0.1, 0.2], num_inner_iterations=5)
step2 = SEQUOIA_Solution_step([1.1, 2.1], 0.45, [0.05, 0.15], num_inner_iterations=nothing)
add_step!(iterates, step1)
add_step!(iterates, step2)

all_inner_iterations = get_all_inner_iterations(iterates)
"""
function get_all_inner_iterations(iterates::SEQUOIA_Iterates)
    [step.num_inner_iterations for step in iterates.steps]
end

"""
    get_all_inner_comp_times(iterates::SEQUOIA_Iterates) -> Vector{Float64}

Return a vector of all inner computation times (`inner_comp_time`) from the stored steps in `SEQUOIA_Iterates`.

# Example

```julia
iterates = SEQUOIA_Iterates()
step1 = SEQUOIA_Solution_step([1.0, 2.0], 0.5, [0.1, 0.2], inner_comp_time=0.02)
step2 = SEQUOIA_Solution_step([1.1, 2.1], 0.45, [0.05, 0.15], inner_comp_time=0.03)
add_step!(iterates, step1)
add_step!(iterates, step2)

all_comp_times = get_all_inner_comp_times(iterates)
"""
function get_all_inner_comp_times(iterates::SEQUOIA_Iterates)
    [step.inner_comp_time for step in iterates.steps]
end

"""
    get_all_solver_statuses(iterates::SEQUOIA_Iterates) -> Vector{SolverStatus}

Return a vector of all solver statuses (`solver_status`) from the stored steps in `SEQUOIA_Iterates`.

# Example

```julia
iterates = SEQUOIA_Iterates()
step1 = SEQUOIA_Solution_step([1.0, 2.0], 0.5, [0.1, 0.2], solver_status=:success)
step2 = SEQUOIA_Solution_step([1.1, 2.1], 0.45, [0.05, 0.15], solver_status=:failed)
add_step!(iterates, step1)
add_step!(iterates, step2)

all_statuses = get_all_solver_statuses(iterates)
"""
function get_all_solver_statuses(iterates::SEQUOIA_Iterates)
    [step.solver_status for step in iterates.steps]
end

