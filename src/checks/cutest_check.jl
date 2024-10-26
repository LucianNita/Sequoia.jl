"""
    validate_cutest_to_sequoia(pb::SEQUOIA_pb, cutest_problem::CUTEstModel)

Main validation function that checks if a `SEQUOIA_pb` instance is correctly initialized 
from a `CUTEstModel` problem. This function ensures that all fields of the `SEQUOIA_pb` object 
are consistent with the CUTEst problem data.

# Arguments
- `pb`: The `SEQUOIA_pb` instance to validate.
- `cutest_problem`: The original `CUTEstModel` instance used for validation.

# Throws
- `ArgumentError` if any field of the `SEQUOIA_pb` instance does not match the expected CUTEst data.
"""
function validate_cutest_to_sequoia(pb::SEQUOIA_pb, cutest_problem::CUTEstModel)
    validate_cutest_nvar(pb, cutest_problem)
    validate_cutest_x0(pb, cutest_problem)
    validate_cutest_objective(pb, cutest_problem)
    validate_cutest_gradient(pb, cutest_problem)
    validate_cutest_constraints(pb, cutest_problem)
    validate_cutest_jacobian(pb, cutest_problem)
end

"""
    validate_cutest_nvar(pb::SEQUOIA_pb, cutest_problem::CUTEstModel)

Validates that the number of variables in the `SEQUOIA_pb` instance matches the `CUTEstModel`.

# Arguments
- `pb`: The `SEQUOIA_pb` instance to validate.
- `cutest_problem`: The original `CUTEstModel` instance for reference.

# Throws
- `ArgumentError` if the number of variables does not match.
"""
function validate_cutest_nvar(pb::SEQUOIA_pb, cutest_problem::CUTEstModel)
    if pb.nvar != cutest_problem.meta.nvar
        throw(ArgumentError("Number of variables `nvar` in `SEQUOIA_pb` does not match the `CUTEstModel`. Expected $(cutest_problem.meta.nvar), got $(pb.nvar)."))
    end
end

"""
    validate_cutest_x0(pb::SEQUOIA_pb, cutest_problem::CUTEstModel)

Validates that the initial guess `x0` in the `SEQUOIA_pb` instance matches the `CUTEstModel`.

# Arguments
- `pb`: The `SEQUOIA_pb` instance to validate.
- `cutest_problem`: The original `CUTEstModel` instance for reference.

# Throws
- `ArgumentError` if the initial guess does not match.
"""
function validate_cutest_x0(pb::SEQUOIA_pb, cutest_problem::CUTEstModel)
    if length(pb.x0) != length(cutest_problem.meta.x0) || !all(pb.x0 .≈ cutest_problem.meta.x0)
        throw(ArgumentError("Initial guess `x0` in `SEQUOIA_pb` does not match the `CUTEstModel`."))
    end
end

"""
    validate_cutest_objective(pb::SEQUOIA_pb, cutest_problem::CUTEstModel)

Validates that the objective function in the `SEQUOIA_pb` instance is correctly defined using the `CUTEstModel`.

# Arguments
- `pb`: The `SEQUOIA_pb` instance to validate.
- `cutest_problem`: The original `CUTEstModel` instance for reference.

# Throws
- `ArgumentError` if the objective function output does not match the expected value.
"""
function validate_cutest_objective(pb::SEQUOIA_pb, cutest_problem::CUTEstModel)
    x_test = pb.x0  # Use the initial guess for testing
    expected_value = obj(cutest_problem, x_test)
    if pb.objective(x_test) != expected_value
        throw(ArgumentError("Objective function in `SEQUOIA_pb` does not match the `CUTEstModel`."))
    end
end

"""
    validate_cutest_gradient(pb::SEQUOIA_pb, cutest_problem::CUTEstModel)

Validates that the gradient function in the `SEQUOIA_pb` instance matches the gradient from the `CUTEstModel`.

# Arguments
- `pb`: The `SEQUOIA_pb` instance to validate.
- `cutest_problem`: The original `CUTEstModel` instance for reference.

# Throws
- `ArgumentError` if the gradient does not match the expected value.
"""
function validate_cutest_gradient(pb::SEQUOIA_pb, cutest_problem::CUTEstModel)
    x_test = pb.x0  # Use the initial guess for testing
    expected_gradient = grad(cutest_problem, x_test)
    computed_gradient = pb.gradient(x_test)

    if !all(computed_gradient .≈ expected_gradient)
        throw(ArgumentError("Gradient in `SEQUOIA_pb` does not match the `CUTEstModel`."))
    end
end

"""
    validate_cutest_constraints(pb::SEQUOIA_pb, cutest_problem::CUTEstModel)

Validates that the constraints function in the `SEQUOIA_pb` instance is correctly defined using the `CUTEstModel`.

# Arguments
- `pb`: The `SEQUOIA_pb` instance to validate.
- `cutest_problem`: The original `CUTEstModel` instance for reference.

# Throws
- `ArgumentError` if the constraints do not match the expected value.
"""
function validate_cutest_constraints(pb::SEQUOIA_pb, cutest_problem::CUTEstModel)
    if pb.constraints !== nothing
        x_test = pb.x0  # Use the initial guess for testing
        expected_constraints = cons(cutest_problem, x_test)
        computed_constraints = pb.constraints(x_test)

        if !all(computed_constraints .≈ expected_constraints)
            throw(ArgumentError("Constraints in `SEQUOIA_pb` do not match the `CUTEstModel`."))
        end
    end
end

"""
    validate_cutest_jacobian(pb::SEQUOIA_pb, cutest_problem::CUTEstModel)

Validates that the Jacobian in the `SEQUOIA_pb` instance matches the Jacobian from the `CUTEstModel`.

# Arguments
- `pb`: The `SEQUOIA_pb` instance to validate.
- `cutest_problem`: The original `CUTEstModel` instance for reference.

# Throws
- `ArgumentError` if the Jacobian does not match the expected value.
"""
function validate_cutest_jacobian(pb::SEQUOIA_pb, cutest_problem::CUTEstModel)
    if pb.jacobian !== nothing
        x_test = pb.x0  # Use the initial guess for testing
        expected_jacobian = jac(cutest_problem, x_test)
        computed_jacobian = pb.jacobian(x_test)

        if !all(computed_jacobian .≈ expected_jacobian)
            throw(ArgumentError("Jacobian in `SEQUOIA_pb` does not match the `CUTEstModel`."))
        end
    end
end