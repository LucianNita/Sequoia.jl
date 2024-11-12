export SEQUOIA_Solution_step

# Define solver statuses as a list of allowed symbols
const SolverStatus = [
    :first_order,     # Satisfied first-order optimality conditions
    :acceptable,      # Convergence is acceptable but not optimal
    :max_iter,        # Reached the maximum number of iterations
    :max_time,        # Exceeded the maximum allowed computation time
    :unbounded,       # The problem appears to be unbounded
    :infeasible,      # The problem appears to be infeasible
    :small_residual,  # Residual norm is below a small threshold
    :small_step,      # Step size is below a small threshold
    :unknown          # Solver status is unknown or not classified
]

"""
    SEQUOIA_Solution_step

Represents the state of a solution at a particular iteration in the SEQUOIA optimization solver. This structure 
captures detailed information about the solver's state, including the solution vector, metrics, constraints, 
and status.

# Fields

- `outer_iteration_number::Int`: The current outer iteration number.
- `convergence_metric::Float64`: A metric used to measure convergence (e.g., feasibility, gradient norm, or optimality gap).
- `solver_status::Symbol`: The status of the solver after this step. Typical values include:
    - `:first_order`: Satisfied first-order optimality conditions.
    - `:acceptable`: Satisfied acceptable convergence tolerance.
    - `:max_iter`: Reached the maximum number of iterations.
    - `:max_time`: Exceeded the maximum allowed computation time.
    - `:unbounded`: Problem appears unbounded.
    - `:infeasible`: Problem appears infeasible.
    - `:small_residual`: Residual norm is below a small threshold.
    - `:small_step`: Step size is below a small threshold.
    - `:unknown`: Solver status is unknown.
- `inner_comp_time::Float64`: The time elapsed during the inner solve call, measured in seconds.
- `num_inner_iterations::Int`: The number of inner iterations completed before exiting.
- `x::Vector{Float64}`: The solution vector at this iteration.
- `fval::Float64`: The value of the objective function at this iteration.
- `gval::Vector{Float64}`: The gradient of the objective function at this iteration.
- `cval::Union{Nothing, Vector{Float64}}`: The constraints evaluated at this iteration, or `nothing` if no constraints.
- `solver_params::Union{Nothing, Vector{Float64}}`: Optional solver-related parameters, such as Lagrange multipliers, penalty parameters, or warm-start values.
- `x_iterates::Union{Nothing, Vector{Vector{Float64}}}`: A vector storing the history of all inner `x` iterates during this step, or `nothing` if not stored.


# Constructor

## Full Constructor
    SEQUOIA_Solution_step(outer_iteration_number::Int,
                          convergence_metric::Float64,
                          solver_status::Symbol,
                          inner_comp_time::Float64,
                          num_inner_iterations::Int,
                          x::Vector{Float64},
                          fval::Float64,
                          gval::Vector{Float64},
                          cval::Union{Nothing, Vector{Float64}} = nothing,
                          solver_params::Union{Nothing, Vector{Float64}} = nothing,
                          x_iterates::Union{Nothing, Vector{Vector{Float64}}} = nothing)

Creates an instance of `SEQUOIA_Solution_step` with all relevant information about a solver step.

# Arguments
- `outer_iteration_number::Int`: The current outer iteration number.
- `convergence_metric::Float64`: A metric used to measure convergence (e.g., feasibility, gradient norm, or optimality gap).
- `solver_status::Symbol`: The status of the solver after this step (e.g., `:success`, `:failed`, etc.).
- `inner_comp_time::Float64`: Time elapsed during the inner solve call, in seconds.
- `num_inner_iterations::Int`: Number of inner iterations completed before exit.
- `x::Vector{Float64}`: The solution vector at this iteration.
- `fval::Float64`: Objective function value at this iteration.
- `gval::Vector{Float64}`: Gradient vector at this iteration.
- `cval::Union{Nothing, Vector{Float64}}`: (Optional) Constraints evaluated at this iteration. Defaults to `nothing`.
- `solver_params::Union{Nothing, Vector{Float64}}`: (Optional) Solver-related parameters (e.g., penalty parameters). Defaults to `nothing`.
- `x_iterates::Union{Nothing, Vector{Vector{Float64}}}`: (Optional) History of all inner `x` iterates. Defaults to `nothing`.

# Returns
A `SEQUOIA_Solution_step` instance.
"""
mutable struct SEQUOIA_Solution_step
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

    """
    Constructor for SEQUOIA_Solution_step.
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
        # Return the solution instance
        return new(outer_iteration_number, convergence_metric, solver_status, inner_comp_time, num_inner_iterations, x, fval, gval, cval, solver_params, x_iterates)
    end
end

