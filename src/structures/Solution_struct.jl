export SEQUOIA_Solution_step

# Define solver statuses as a list of allowed symbols
SolverStatus = [
    :first_order,
    :acceptable,
    :max_iter,
    :max_time,
    :unbounded,
    :infeasible,
    :small_residual,
    :small_step,
    :stalled,
    :unknown,
    :user
    #:invalid_start,
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
    solver_params::Union{Nothing, Vector{Float64}}                  # Optional solver-related parameters. Used for warm starting. Can be lagrange multipliers, penalty parameter, objective upper bound etc.
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
        #validate_sequoia_solution!(solution) #For testing only, comment-out when releasing

        # Return the valid solution
        return solution
    end
end

