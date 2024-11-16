export r, r_gradient!

"""
    r(x, tk, problem::CUTEstModel)

Compute the penalty function r(x, t_k) for a `CUTEstModel` problem.

# Arguments
- `x`: The vector of decision variables.
- `tk`: The threshold value for the objective function.
- `problem`: A `CUTEstModel` problem instance.

# Returns
- A scalar value representing the penalty function:
  r(x, t_k) = r_0(x) + 0.5 * max(0, f(x) - t_k)^2,
  where:
  - `r_0(x)` is the constraint violation penalty.
  - max(0, f(x) - t_k) penalizes the objective for exceeding the threshold.

# Notes
- `r_0(x)` penalizes equality and inequality constraint violations.
- The objective penalty applies only when f(x) > t_k.
"""
function r(x,tk,problem::CUTEstModel)
    cviol = r0(x,problem);
    obj_penalty = max(0.0, obj(problem, x)-tk);

    return cviol + 0.5 * obj_penalty^2; 
end

"""
    r_gradient!(grad_storage, x, tk, problem::CUTEstModel)

Compute the gradient of the penalty function r(x, t_k) for a `CUTEstModel` problem and store it in `grad_storage`.

# Arguments
- `grad_storage`: A preallocated vector to store the gradient result.
- `x`: The vector of decision variables.
- `tk`: The threshold value for the objective function.
- `problem`: A `CUTEstModel` problem instance.

# Notes
- The gradient is computed as:
  ∇r(x, t_k) = ∇r_0(x) + ∇f(x) * max(0, f(x) - t_k),
  where:
  - ∇r_0(x) is the gradient of the constraint violation penalty.
  - ∇f(x) is the gradient of the objective function.
- The `grad_storage` vector is updated in place.
"""
function r_gradient!(grad_storage, x, tk, problem::CUTEstModel)
    r0_gradient!(grad_storage,x,problem);
    grad_storage .= grad_storage .+ (grad(problem, x) .* max(0.0, obj(problem,x) - tk));# (obj(problem,x)>=tk)
end

"""
    r(x, tk, problem::SEQUOIA_pb)

Compute the penalty function r(x, t_k) for a `SEQUOIA_pb` problem.

# Arguments
- `x`: The vector of decision variables.
- `tk`: The threshold value for the objective function.
- `problem`: A `SEQUOIA_pb` problem instance.

# Returns
- A scalar value representing the penalty function:
  r(x, t_k) = r_0(x) + 0.5 * max(0, f(x) - t_k)^2,
  where:
  - `r_0(x)` is the constraint violation penalty.
  - max(0, f(x) - t_k) penalizes the objective for exceeding the threshold.

# Notes
- `r_0(x)` penalizes equality and inequality constraint violations.
- The objective penalty applies only when f(x) > t_k.
"""
function r(x,tk,problem::SEQUOIA_pb)
    cviol = r0(x,problem);
    obj_penalty = max(0.0, problem.objective(x)-tk);

    return cviol + 0.5 * obj_penalty^2;
end

"""
    r_gradient!(grad_storage, x, tk, problem::SEQUOIA_pb)

Compute the gradient of the penalty function r(x, t_k) for a `SEQUOIA_pb` problem and store it in `grad_storage`.

# Arguments
- `grad_storage`: A preallocated vector to store the gradient result.
- `x`: The vector of decision variables.
- `tk`: The threshold value for the objective function.
- `problem`: A `SEQUOIA_pb` problem instance.

# Notes
- The gradient is computed as:
  ∇r(x, t_k) = ∇r_0(x) + ∇f(x) * max(0, f(x) - t_k),
  where:
  - ∇r_0(x) is the gradient of the constraint violation penalty.
  - ∇f(x) is the gradient of the objective function.
- The `grad_storage` vector is updated in place.
"""
function r_gradient!(grad_storage, x, tk, problem::SEQUOIA_pb)
    r0_gradient!(grad_storage,x,problem);
    grad_storage .= grad_storage .+ (problem.gradient(x) .* max(0.0, problem.objective(x) - tk)); ##(problem.objective(x)>=tk)
end