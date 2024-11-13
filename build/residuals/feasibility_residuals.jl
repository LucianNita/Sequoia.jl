export r0, r0_gradient!

"""
    r0(x, problem::CUTEstModel)

Compute the residual function r_0(x) for a `CUTEstModel` problem.

# Arguments
- `x`: The vector of variables.
- `problem::CUTEstModel`: A CUTEst optimization problem.

# Returns
- A scalar value representing the residual function, calculated as:
    r_0(x) = 1/2 * (sum(con[i]^2 for i in eqcon) + sum(max(0, con[j])^2 for j in ineqcon))
    where:
    - Equality constraints (`eqcon`) include the first `jfix + ifix` constraints.
    - Inequality constraints (`ineqcon`) are the remaining constraints.

# Notes
- The function computes penalties for equality constraints (`eqcon`) using squared residuals.
- Inequality constraints (`ineqcon`) are only penalized for violations (using a ReLU-like quadratic penalty).
"""
function r0(x,problem::CUTEstModel)
    con = res(x,problem);
    total_eq_con = length(problem.meta.jfix)+length(problem.meta.ifix);
    return 0.5*(sum(con[1:total_eq_con].^2)+sum( (max.(0,con[total_eq_con+1:end])).^2 ))
end

"""
    r0_gradient!(g, x, problem::CUTEstModel)

Compute the gradient of the residual function ∇r_0(x) for a `CUTEstModel` problem and store it in `g`.

# Arguments
- `g`: A preallocated vector to store the gradient. Must have the same size as `x`.
- `x`: The vector of variables.
- `problem::CUTEstModel`: A CUTEst optimization problem.

# Notes
- The gradient is computed as:
    g = J' * con
    where:
    - `J` is the Jacobian of the constraints.
    - `con` includes squared penalties for equality constraints and ReLU penalties for inequality constraints.
"""
function r0_gradient!(g, x, problem::CUTEstModel)
    # Compute residuals
    con = res(x, problem)
    total_eq_con = length(problem.meta.jfix)+length(problem.meta.ifix);

    # Apply max(0, ...) for inequality constraints
    con[total_eq_con+1:end] .= max.(0, con[total_eq_con+1:end])

    # Compute the Jacobian
    J = dresdx(x, problem)

    # In-place multiplication for gradient computation
    mul!(g, J', con)
end

"""
    r0(x, problem::SEQUOIA_pb)

Compute the residual function ( r_0(x) ) for a `SEQUOIA_pb` problem.

# Arguments
- `x`: The vector of variables.
- `problem::SEQUOIA_pb`: A SEQUOIA optimization problem.

# Returns
- A scalar value representing the residual function, calculated as:
r_0(x) = 1/2 * (sum(con[i]^2 for i in eqcon) + sum(max(0, con[j])^2 for j in ineqcon))

# Notes
- The function computes penalties for equality constraints (`eqcon`) and inequality constraints (`ineqcon`) using squared residuals.
"""
function r0(x,problem::SEQUOIA_pb)
    constraint_val = problem.constraints(x);
    return 0.5 * ( sum( (constraint_val[problem.eqcon]).^2 ) + sum( (max.(0.0, constraint_val[problem.ineqcon])).^2 ) )
end

"""
    r0_gradient!(g, x, problem::SEQUOIA_pb)

Compute the gradient of the residual function ∇r_0(x) for a `SEQUOIA_pb` problem and store it in `g`.

# Arguments
- `g`: A preallocated vector to store the gradient. Must have the same size as `x`.
- `x`: The vector of variables.
- `problem::SEQUOIA_pb`: A SEQUOIA optimization problem.

# Notes
- The gradient is computed as:
    g = J' * con
    where:
    - `J` is the Jacobian of the constraints.
    - `constraint_val` includes penalties for equality and inequality constraints.
"""
function r0_gradient!(g, x, problem::SEQUOIA_pb)
    # Compute residuals
    constraint_val = problem.constraints(x)

    # Apply max(0, ...) for inequality constraints
    constraint_val[problem.ineqcon] = max.(0,constraint_val[problem.ineqcon])

    # Compute the Jacobian
    jac = problem.jacobian(x)

    # In-place multiplication for gradient computation
    mul!(g, jac', constraint_val)
end