using ForwardDiff

export validate_pb,validate_constraints!

"""
    validate_pb(pb::SEQUOIA_pb)

Main validation function for a `SEQUOIA_pb` optimization problem. Ensures all required components of the problem 
are correctly defined, including the number of variables, initial guess, objective function, gradient, constraints, 
and solver settings.

# Arguments

- `pb`: The `SEQUOIA_pb` problem instance to validate.

# Throws

- `ArgumentError` if any of the fields or functions in the problem are incorrectly defined or inconsistent.
"""
function validate_pb(pb::SEQUOIA_pb) # Main validation function for SEQUOIA_pb
    # Validate the number of variables
    validate_nvar(pb.nvar)
    
    # Validate the initial guess
    validate_x0(pb.x0, pb.nvar)

    # Validate the objective function
    validate_objective(pb.objective, pb.x0)

    # Validate the gradient function (with fallback to automatic differentiation if not set)
    validate_gradient!(pb)

    # Validate the constraints function and its consistency with the specified equality/inequality indices
    validate_constraints!(pb)

    # Validate exit code
    validate_code(pb.exitCode)

end

"""
    validate_nvar(nvar::Int)

Validates that the number of variables `nvar` is a positive integer.

# Arguments

- `nvar`: The number of variables to validate.

# Throws

- `ArgumentError` if `nvar` is not a positive integer.
"""
function validate_nvar(nvar::Int)
    if nvar <= 0
        throw(ArgumentError("The number of variables `nvar` must be a positive integer."))
    end
end

"""
    validate_x0(x0::Vector{Float64}, nvar::Int)

Validates that the length of the initial guess vector `x0` matches the number of variables `nvar`.

# Arguments

- `x0`: The initial guess vector.
- `nvar`: The expected number of variables.

# Throws

- `ArgumentError` if the length of `x0` does not match `nvar`.
"""
function validate_x0(x0::Vector{Float64}, nvar::Int)
    if length(x0) != nvar
        throw(ArgumentError("The length of the initial guess `x0` must be equal to `nvar`."))
    end
end


"""
    validate_objective(objective::Union{Nothing, Function}, x0::Vector{Float64})

Validates that the objective function is callable and returns a scalar of type `Float64`. 
Throws an error if the objective is not defined or returns invalid output.

# Arguments

- `objective`: The objective function to validate.
- `x0`: The initial guess vector to test the objective function.

# Throws

- `ArgumentError` if the objective is not defined or does not return a scalar `Float64`.
"""
function validate_objective(objective::Union{Nothing, Function}, x0::Vector{Float64})
    if objective === nothing
        throw(ArgumentError("An objective function is required. Please set one using `set_objective!(problem::SEQUOIA_pb)`."))
    end

    # Test the objective function with the initial guess to check its output
    result = objective(x0)
    
    if !(isa(result, Float64))
        throw(ArgumentError("The objective function must return a scalar of type `Float64`. Multi-objective optimization is not supported."))
    end
end

"""
    validate_gradient!(pb::SEQUOIA_pb)

Validates that the gradient function is callable and returns a vector of length `nvar`. 
If no gradient is provided, automatic differentiation using `ForwardDiff` is used as a fallback.

# Arguments

- `pb`: The `SEQUOIA_pb` problem instance.

# Throws

- `ArgumentError` if the gradient function does not return a vector of `Float64` of the correct size.
"""
function validate_gradient!(pb::SEQUOIA_pb)
    if pb.gradient === nothing
        @warn "A gradient is required. Setting one using Automatic Differentiation with ForwardDiff."
        pb.gradient = x -> ForwardDiff.gradient(pb.objective, x)
    end

    # Test the gradient function with the initial guess to check its output
    result = pb.gradient(pb.x0)

    # Check if the result is a vector of Float64 and matches the size of `nvar`
    if !(isa(result, Vector{Float64}) && length(result) == pb.nvar)
        throw(ArgumentError("The gradient function must return a vector of Float64 of size `nvar`."))
    end
end

"""
    validate_constraints!(pb::SEQUOIA_pb)

Validates the constraints function if provided, ensuring consistency with the specified equality and inequality constraints. 
If no constraints are provided, a warning is issued. Also validates the Jacobian if needed.

# Arguments

- `pb`: The `SEQUOIA_pb` problem instance.

# Throws

- `ArgumentError` if the constraints are inconsistent with the equality/inequality indices or if the Jacobian is invalid.
"""
function validate_constraints!(pb::SEQUOIA_pb)
    if pb.constraints === nothing 
        @warn "No constraints are set. Ensure this is intended, as SEQUOIA is tailored for constrained optimization."
    end

    if pb.constraints !== nothing 
        # Ensure the number of constraints matches the number of specified indices
        num_constraints = length(pb.constraints(pb.x0))
        total_specified_constraints = length(pb.eqcon) + length(pb.ineqcon)
        if num_constraints != total_specified_constraints
            throw(ArgumentError("The number of constraints returned by the constraint function does not match the total number of specified constraints (equality + inequality)."))
        end

        # Ensure all indices are covered exactly once by eqcon and ineqcon
        all_indices = sort(vcat(pb.eqcon, pb.ineqcon))
        if all_indices != collect(1:num_constraints)
            throw(ArgumentError("Indices for equality and inequality constraints must cover all constraint indices exactly once."))
        end

        # Validate the Jacobian
        validate_jacobian!(pb)
    end
end

"""
    validate_jacobian!(pb::SEQUOIA_pb)

Validates the Jacobian function if provided. If no Jacobian is defined, automatic differentiation with `ForwardDiff` is used as a fallback.

# Arguments

- `pb`: The `SEQUOIA_pb` problem instance.

# Throws

- `ArgumentError` if the Jacobian is invalid or does not return the correct matrix size.
"""
function validate_jacobian!(pb::SEQUOIA_pb)
    if pb.jacobian === nothing
        @warn "A Jacobian is required. Setting one using Automatic Differentiation with ForwardDiff."
        pb.jacobian = x -> ForwardDiff.jacobian(pb.constraints, x)
    end

    # Test the jacobian function with the initial guess to check its output
    result = pb.jacobian(pb.x0)
    num_constraints = length(pb.constraints(pb.x0))

    # Check if the result is a matrix of Float64 and has the correct size (num_constraints x nvar)
    if size(result) != (num_constraints, pb.nvar) #|| !isa(result, Matrix{Float64}) 
        throw(ArgumentError("The Jacobian must be a matrix of size (num_constraints, nvar), where num_constraints is the number of constraints and nvar is the number of variables.")) # of Float64
    end
end


"""
    validate_code(code::Symbol)

Validates the `exitCode` to ensure it is a valid symbol from the predefined list of exit codes.

# Arguments

- `code`: The exit code symbol to validate.

# Throws

- `ArgumentError` if the `code` is not in the predefined list `ExitCode`.
"""
function validate_code(code::Symbol)
    if !(code in ExitCode)
        throw(ArgumentError("Invalid exit code: `$code`. Must be one of: $ExitCode."))
    end
end

"""
    objective_setter_fallback(objective::Any)

Validates that the provided objective function is callable (i.e., a function). 
If the input is not a callable function, an error is raised.

# Arguments

- `objective`: The object to validate as an objective function.

# Throws

- `ArgumentError` if the `objective` is not a callable function.
"""
function objective_setter_fallback(objective::Any)
    if !isa(objective, Function)
        throw(ArgumentError("The objective must be a callable function."))
    end
end

"""
    solver_settings_fallback(solver_settings::Any)

Validates that the solver settings are of type `SEQUOIA_Settings`.

# Arguments

- `solver_settings`: The solver settings to validate.

# Throws

- `ArgumentError` if `solver_settings` is not of type `SEQUOIA_Settings`.
"""
function solver_settings_fallback(solver_settings::Any)
    if !(solver_settings isa SEQUOIA_Settings)
        throw(ArgumentError("`solver_settings` must be of type `SEQUOIA_Settings`."))
    end
end

"""
    pb_fallback(pb::Any)

Validates that the problem instance `pb` is of type `SEQUOIA_pb`. If it is not, throws an `ArgumentError`. This function is primarily used internally to ensure that the input to SEQUOIA-related functions is a valid `SEQUOIA_pb` instance.

# Arguments
- `pb`: The problem instance to validate.

# Throws
- `ArgumentError`: If `pb` is not of type `SEQUOIA_pb`.
"""
function pb_fallback(pb::Any)
    if !(pb isa SEQUOIA_pb)
        throw(ArgumentError("Problem `pb` must be of type `SEQUOIA_pb`."))
    end
end
