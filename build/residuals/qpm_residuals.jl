export qpm_obj, qpm_grad!
"""
    qpm_obj(x, μ, problem::SEQUOIA_pb)

Compute the quadratic penalty method objective function.

# Arguments
- `x`: The decision variable vector.
- `μ`: The penalty parameter (scalar).
- `problem`: An optimization problem of type `SEQUOIA_pb`.

# Returns
- The penalized objective value: `f(x) + μ * r_0(x)`.

# Notes
- For `SEQUOIA_pb`, `problem.objective` and `problem.constraints` are used.
"""
function qpm_obj(x,μ,problem::SEQUOIA_pb)
    return problem.objective(x)+μ*r0(x,problem)
end

"""
    qpm_grad!(g, x, μ, problem::SEQUOIA_pb)

Compute the gradient of the quadratic penalty method objective function and store it in `g`.

# Arguments
- `g`: A preallocated gradient vector.
- `x`: The decision variable vector.
- `μ`: The penalty parameter (scalar).
- `problem`: An optimization problem of type `SEQUOIA_pb`.

# Notes
- For `SEQUOIA_pb`, `problem.gradient` and `problem.jacobian` are used.
"""
function qpm_grad!(g, x, μ, problem::SEQUOIA_pb)
    grad_obj = problem.gradient(x); # Compute the gradient of the base objective
    r0_gradient!(g,x,problem); # Compute the gradient of the penalty term
    g .=  grad_obj .+ μ .* g; # Update the gradient with the penalty term
end

"""
    qpm_obj(x, μ, problem::CUTEstModel)

Compute the quadratic penalty method objective function.

# Arguments
- `x`: The decision variable vector.
- `μ`: The penalty parameter (scalar).
- `problem`: An optimization problem of type `CUTEstModel`.

# Returns
- The penalized objective value: `f(x) + μ * r_0(x)`.

# Notes
- For `CUTEstModel`, `obj` and `res` are used.
"""
function qpm_obj(x,μ,problem::CUTEstModel)
    return obj(problem,x)+μ*r0(x,problem)
end

"""
    qpm_grad!(g, x, μ, problem::CUTEstModel)

Compute the gradient of the quadratic penalty method objective function and store it in `g`.

# Arguments
- `g`: A preallocated gradient vector.
- `x`: The decision variable vector.
- `μ`: The penalty parameter (scalar).
- `problem`: An optimization problem of type `CUTEstModel`.

# Notes
- For `CUTEstModel`, `grad` and `dresdx` are used.
"""
function qpm_grad!(g, x, μ, problem::CUTEstModel)
    grad_obj = grad(problem,x); # Compute the gradient of the base objective
    r0_gradient!(g,x,problem); # Compute the gradient of the penalty term
    g .=  grad_obj .+ μ .* g; # Update the gradient with the penalty term
end
