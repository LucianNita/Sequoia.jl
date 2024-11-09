export validate_sequoia_solution! 

"""
    validate_sequoia_solution!(solution::SEQUOIA_Solution_step)

Validates the fields of a `SEQUOIA_Solution_step` instance. This function ensures the correctness and consistency 
of the solution vector, gradients, constraints, solver parameters, and metadata.

# Arguments
- `solution`: An instance of `SEQUOIA_Solution_step` to validate.

# Throws
- `ArgumentError` if any of the validation checks fail.
"""
function validate_sequoia_solution!(solution::SEQUOIA_Solution_step)
    # Validate `outer_iteration_number`
    if solution.outer_iteration_number < 0
        throw(ArgumentError("`outer_iteration_number` must be a non-negative integer. Got: $(solution.outer_iteration_number)."))
    end

    # Validate `convergence_metric`
    if solution.convergence_metric < 0
        throw(ArgumentError("`convergence_metric` must be non-negative. Got: $(solution.convergence_metric)."))
    end

    # Validate `solver_status`
    validate_solver_status(solution.solver_status)

    # Validate `inner_comp_time`
    if solution.inner_comp_time < 0
        throw(ArgumentError("`inner_comp_time` must be non-negative. Got: $(solution.inner_comp_time)."))
    end

    # Validate `num_inner_iterations`
    if solution.num_inner_iterations < 0
        throw(ArgumentError("`num_inner_iterations` must be a non-negative integer. Got: $(solution.num_inner_iterations)."))
    end

    # Validate `x` (solution vector)
    if isempty(solution.x)
        throw(ArgumentError("Solution vector `x` cannot be empty."))
    end

    # Validate `gval` (gradient vector)
    if isempty(solution.gval)
        throw(ArgumentError("Gradient vector `gval` cannot be empty."))
    end
    if length(solution.x) != length(solution.gval)
        throw(ArgumentError("`gval` must have the same number of elements as `x`. Got: $(length(solution.gval)) vs $(length(solution.x))."))
    end

    # Validate `fval` (objective function value)
    if !(solution.fval isa Float64)
        throw(ArgumentError("Objective function value `fval` must be a `Float64`. Got: $(typeof(solution.fval))."))
    end

    # Validate `cval` (constraints vector) if provided
    if !isnothing(solution.cval)
        if !all(isa(c, Float64) for c in solution.cval)
            throw(ArgumentError("All elements in `cval` must be `Float64`. Got: $(solution.cval)."))
        end
    end

    # Validate `solver_params` if provided
    if !isnothing(solution.solver_params)
        if !all(isa(p, Float64) for p in solution.solver_params)
            throw(ArgumentError("`solver_params` must be a vector of `Float64`. Got: $(solution.solver_params)."))
        end
    end

    # Validate `x_iterates` if provided
    if !isnothing(solution.x_iterates)
        if length(solution.x_iterates) != solution.num_inner_iterations + 1
            throw(ArgumentError("`x_iterates` must contain exactly `num_inner_iterations + 1` vectors. Got: $(length(solution.x_iterates))."))
        end
        for iter in solution.x_iterates
            if length(iter) != length(solution.x)
                throw(ArgumentError("Each vector in `x_iterates` must have the same number of elements as `x`. Got: $(length(iter)) vs $(length(solution.x))."))
            end
        end
    end
end

"""
    validate_solver_status(status::Symbol)

Checks if the solver status is valid by ensuring it is one of the allowed symbols in `SolverStatus`.
Throws an `ArgumentError` if the status is not valid.
"""
function validate_solver_status(status::Symbol)
    if !(status in SolverStatus)
        throw(ArgumentError("Invalid `solver_status`: $status. Valid statuses are: $(join(SolverStatus, ", "))."))
    end
end