"""
    auglag_obj(x, μ, λ, problem::SEQUOIA_pb)

Compute the augmented Lagrangian objective for a `SEQUOIA_pb` problem.

# Arguments
- `x`: The vector of decision variables.
- `μ`: The penalty parameter (scalar).
- `λ`: The vector of Lagrange multipliers for constraints.
- `problem`: A `SEQUOIA_pb` problem instance.

# Returns
- A scalar value representing the augmented Lagrangian objective, calculated as:
  `f(x) + μ * r₀(x) + λᵀ * g(x)`
  where:
  - `f(x)` is the original objective function.
  - `r₀(x)` is the quadratic penalty for constraints.
  - `g(x)` is the constraint violation vector, with `max(0, g[j])` applied to inequality constraints.

# Notes
- Inequality constraints (`ineqcon`) are penalized only if violated (using a ReLU-like max operation).
"""
function auglag_obj(x, μ, λ, problem::SEQUOIA_pb)
    
    constraint_val = problem.constraints(x);
    constraint_val[problem.ineqcon] = max.(0,constraint_val[problem.ineqcon]); 

    return problem.objective(x) + μ*r0(x,problem) + constraint_val' * λ
end

"""
    auglag_grad!(g, x, μ, λ, problem::SEQUOIA_pb)

Compute the gradient of the augmented Lagrangian objective for a `SEQUOIA_pb` problem and store it in `g`.

# Arguments
- `g`: A preallocated gradient vector to store the result.
- `x`: The vector of decision variables.
- `μ`: The penalty parameter (scalar).
- `λ`: The vector of Lagrange multipliers for constraints.
- `problem`: A `SEQUOIA_pb` problem instance.

# Notes
- The gradient is calculated as:
  `∇f(x) + μ * ∇r₀(x) + ∇g(x)ᵀ * λ`
  where:
  - `∇f(x)` is the gradient of the objective function.
  - `∇r₀(x)` is the gradient of the quadratic penalty term.
  - `∇g(x)` is the Jacobian of the constraints.
- Only active inequality constraints contribute to the gradient, using a mask to selectively apply penalties.

# Efficiency
- If the Jacobian is sparse, operations are optimized for sparsity.
"""
function auglag_grad!(g, x, μ, λ, problem::SEQUOIA_pb)
    grad_obj = problem.gradient(x);
    r0_gradient!(g,x,problem);

    jac=problem.jacobian(x);
    constraint_val = problem.constraints(x);

    active_mask = sparse(constraint_val .> 0.0)  # Use a sparse mask for active inequality constraints
    jac_modified = jac .* active_mask            # Modify Jacobian using the mask

    g .= grad_obj .+ μ .* g .+ jac_modified' * λ 
end

"""
    update_lag_mult!(x, μ, λ, problem::SEQUOIA_pb)

Update the Lagrange multipliers for a `SEQUOIA_pb` problem.

# Arguments
- `x`: The vector of decision variables.
- `μ`: The penalty parameter (scalar).
- `λ`: The vector of Lagrange multipliers for constraints (modified in place).
- `problem`: A `SEQUOIA_pb` problem instance.

# Notes
- Equality constraints (`eqcon`) are updated using the formula:
  `λ[i] = λ[i] + μ * g[i]`
- Inequality constraints (`ineqcon`) are updated using:
  `λ[i] = max(0, λ[i] + μ * g[i])`
  where `g[i]` is the constraint violation.

# Efficiency
- The implementation efficiently handles in-place updates for `λ` without additional memory allocation.
"""
function update_lag_mult!(x, μ, λ, problem::SEQUOIA_pb)
    constraint_val = problem.constraints(x)
    λ[problem.eqcon] .= λ[problem.eqcon] .+ μ .* constraint_val[problem.eqcon]
    λ[problem.ineqcon] .= max.(0.0, λ[problem.ineqcon] .+ μ .* constraint_val[problem.ineqcon])
end

"""
    auglag_obj(x, μ, λ, problem::CUTEstModel)

Compute the augmented Lagrangian objective for a `CUTEstModel` problem.

# Arguments
- `x`: The vector of decision variables.
- `μ`: The penalty parameter (scalar).
- `λ`: The vector of Lagrange multipliers for constraints.
- `problem`: A `CUTEstModel` problem instance.

# Returns
- A scalar value representing the augmented Lagrangian objective:
  `f(x) + 0.5 * μ * (sum(con_eq.^2) + sum(max(0, con_ineq).^2)) + λᵀ * con`

# Notes
- `con` is split into equality constraints (`con_eq`) and inequality constraints (`con_ineq`) based on problem metadata.
- `max(0, ...)` is applied to penalize only violated inequality constraints.
"""
function auglag_obj(x, μ, λ, problem::CUTEstModel)
    # Compute residuals
    con = res(x, problem)
    # Determine residual partitions
    total_eq_con = length(problem.meta.jfix) + length(problem.meta.ifix)

    # Compute the augmented Lagrangian objective
    obj_val = obj(problem, x)
    penalty_term = 0.5 * μ * (sum(con[1:total_eq_con].^2) + sum(max.(0, con[total_eq_con+1:end]).^2))
    lagrange_term = λ' * con

    return obj_val + penalty_term + lagrange_term
end

"""
    auglag_grad!(g, x, μ, λ, problem::CUTEstModel)

Compute the gradient of the augmented Lagrangian objective for a `CUTEstModel` problem and store it in `g`.

# Arguments
- `g`: A preallocated gradient vector to store the result.
- `x`: The vector of decision variables.
- `μ`: The penalty parameter (scalar).
- `λ`: The vector of Lagrange multipliers for constraints.
- `problem`: A `CUTEstModel` problem instance.

# Notes
- The gradient is calculated as:
  `∇f(x) + μ * ∇r₀(x) + ∇con(x)ᵀ * λ`
  where:
  - `∇f(x)` is the gradient of the objective function.
  - `∇r₀(x)` is the gradient of the penalty term.
  - `∇con(x)` is the Jacobian of the constraints.
- Only active inequality constraints contribute to the gradient, using a mask to selectively apply penalties.
"""
function auglag_grad!(g, x, μ, λ, problem::CUTEstModel)
    grad_obj = grad(problem, x);
    J = dresdx(x, problem)
    con = res(x, problem)
    r0_gradient!(g,x,problem);
    total_eq_con = length(problem.meta.jfix)+length(problem.meta.ifix);

    con[total_eq_con+1:end] .= max.(0, con[total_eq_con+1:end]) # Apply max(0, ...) for inequality constraints

    active_mask = sparse(con .> 0.0)  # Use a sparse mask for active inequality constraints
    jac_modified = J .* active_mask            # Modify Jacobian using the mask

    g .= grad_obj .+ μ .* g .+ jac_modified'*λ
end

"""
    update_lag_mult!(x, μ, λ, problem::CUTEstModel)

Update the Lagrange multipliers for a `CUTEstModel` problem.

# Arguments
- `x`: The vector of decision variables.
- `μ`: The penalty parameter (scalar).
- `λ`: The vector of Lagrange multipliers for constraints (modified in place).
- `problem`: A `CUTEstModel` problem instance.

# Notes
- Equality constraints are updated using:
  `λ[eq_indices] += μ * con[eq_indices]`
- Inequality constraints are updated using:
  `λ[ineq_indices] = max(0, λ[ineq_indices] + μ * con[ineq_indices])`
"""
function update_lag_mult!(x, μ, λ, problem::CUTEstModel)
    con = res(x, problem)
    total_eq_con = length(problem.meta.jfix) + length(problem.meta.ifix)
    
    λ[1:total_eq_con] .= λ[1:total_eq_con] .+ μ .* con[1:total_eq_con]
    λ[total_eq_con+1:end] .= max.(0.0, λ[total_eq_con+1:end] .+ μ .* con[total_eq_con+1:end])
end

