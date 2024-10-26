export validate_sequoia_solution! #For testing purposes only, should not be exported

"""
    validate_sequoia_solution!(solution::SEQUOIA_Solution_step)

Validates the fields of a `SEQUOIA_Solution_step` instance. This function checks the solution vector, function values, gradients, step sizes, convergence metrics, and solver status for validity.

# Arguments
- `solution`: An instance of `SEQUOIA_Solution_step` to validate.

# Throws
- `ArgumentError` if any of the validation checks fail.
"""
function validate_sequoia_solution!(solution::SEQUOIA_Solution_step)
    # Validate the solution vector `x`
    if isempty(solution.x)
        throw(ArgumentError("Solution vector `x` cannot be empty."))
    end

    # Validate the gradient vector `gval`
    if isempty(solution.gval)
        throw(ArgumentError("Gradient vector `gval` cannot be empty."))
    end

    # Validate that the number of elements in `x` matches the number of elements in `gval`
    if length(solution.x) != length(solution.gval)
        throw(ArgumentError("The number of elements in `x` must match the number of elements in `gval`."))
    end

    # Validate that `fval` is numeric
    if !isnumeric(solution.fval)
        throw(ArgumentError("Objective function value `fval` must be numeric."))
    end

    # Validate that `convergence_metric` is numeric
    if !isnumeric(solution.convergence_metric)
        throw(ArgumentError("Convergence metric `convergence_metric` must be numeric."))
    end

    # Validate `inner_comp_time` to ensure it's non-negative and numeric
    if solution.inner_comp_time < 0
        throw(ArgumentError("Inner computation time `inner_comp_time` must be non-negative."))
    end

    # Validate `outer_iteration_number` to ensure it's non-negative
    if solution.outer_iteration_number < 0
        throw(ArgumentError("Outer iteration number `outer_iteration_number` must be non-negative."))
    end

    # Validate `num_inner_iterations`, ensuring it's non-negative
    if solution.num_inner_iterations < 0
        throw(ArgumentError("Number of inner iterations `num_inner_iterations` must be non-negative."))
    end

    # Validate `cval`, ensuring it contains only numeric values if provided
    if !isnothing(solution.cval) && !all(isnumeric, solution.cval)
        throw(ArgumentError("Constraint values `cval` must be a vector of numeric values if provided."))
    end

    # Validate `x_iterates` if provided: it must be a vector of `num_inner_iterations` vectors of the same size as `x`
    if !isnothing(solution.x_iterates)
        if length(solution.x_iterates) != solution.num_inner_iterations+1
            throw(ArgumentError("`x_iterates` must contain exactly `num_inner_iterations`+1 vectors."))
        end
        if any(length(x_iter) != length(solution.x) for x_iter in solution.x_iterates)
            throw(ArgumentError("Each vector in `x_iterates` must have the same number of elements as `x`."))
        end
    end

    # Validate `solver_status`, ensuring it's valid
    validate_solver_status(solution.solver_status)
end

"""
    isnumeric(x)

A utility function to check if a value is numeric or a vector of numeric values (`Float64`).
"""
isnumeric(x) = x isa Number || (x isa AbstractArray && all(isa(x, Number) for x in x))

"""
    validate_solver_status(status::Symbol)

Checks if the solver status is valid by ensuring it is one of the allowed symbols in `SolverStatus`.
Throws an `ArgumentError` if the status is not valid.
"""
function validate_solver_status(status::Symbol)
    if !(status in SolverStatus)
        throw(ArgumentError("Invalid solver status: $status. Valid statuses are: $(join(SolverStatus, ", "))"))
    end
end
