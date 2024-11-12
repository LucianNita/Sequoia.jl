export ipm_obj, ipm_grad!, r_slack

"""
    ipm_obj(x_a, μ, problem::SEQUOIA_pb)

Compute the Interior Point Method (IPM) objective function for a `SEQUOIA_pb` problem.

# Arguments
- `x_a`: A combined vector of decision variables (`x`), Lagrange multipliers (`λ`), and slack variables (`s`).
- `μ`: The barrier penalty parameter (scalar).
- `problem`: A `SEQUOIA_pb` optimization problem.

# Returns
- A scalar value representing the IPM objective, computed as:
||∇f(x) + J(x)'λ||² + ||c(x) + s²||² + ||2ν .* s² - μ||²
where:
- `f(x)` is the objective function.
- `J(x)` is the Jacobian of the constraints.
- `c(x)` are the constraints.
- `ν` are the inequality multipliers.
- `s` are the slack variables.

# Notes
- The IPM objective combines the KKT conditions with slack variables to enforce inequality constraints.
"""
function ipm_obj(x_a, μ, problem::SEQUOIA_pb)
    
    x=x_a[1:problem.nvar];
    iq=length(problem.ineqcon);
    eq=length(problem.eqcon);
    λ=x_a[problem.nvar+1:problem.nvar+iq+eq];
    ν=λ[problem.ineqcon];
    s=x_a[problem.nvar+iq+eq+1:end];
    jac=problem.jacobian(x);

    constraint_val = problem.constraints(x)
    constraint_val[problem.ineqcon] .= constraint_val[problem.ineqcon] .+ s.^2 
    return norm(problem.gradient(x)+jac'*λ)^2+ sum(constraint_val.^2) + norm(2*ν.*s.^2 .-μ)^2 
end

"""
  ipm_grad!(g, x_a, μ, problem::SEQUOIA_pb)

Compute the gradient of the IPM objective function for a `SEQUOIA_pb` problem.

# Arguments
- `g`: A preallocated vector to store the gradient result.
- `x_a`: A combined vector of decision variables (`x`), Lagrange multipliers (`λ`), and slack variables (`s`).
- `μ`: The barrier penalty parameter (scalar).
- `problem`: A `SEQUOIA_pb` optimization problem.

# Notes
- The gradient is computed in place and stored in `g`.
"""
function ipm_grad!(g, x_a, μ, problem::SEQUOIA_pb)
    ForwardDiff.gradient!(g, z -> ipm_obj(z, μ, problem), x_a)
end  

"""
  ipm_obj(x_a, μ, problem::CUTEstModel)

Compute the Interior Point Method (IPM) objective function for a `CUTEstModel` problem.

# Arguments
- `x_a`: A combined vector of decision variables (`x`), Lagrange multipliers (`λ`), and slack variables (`s`).
- `μ`: The barrier penalty parameter (scalar).
- `problem`: A `CUTEstModel` optimization problem.

# Returns
- A scalar value representing the IPM objective, computed similarly to the `SEQUOIA_pb` case but using `CUTEstModel` functions.
"""
function ipm_obj(x_a, μ, problem::CUTEstModel)
    x=x_a[1:problem.meta.nvar];
    eq = length(problem.meta.jfix) + length(problem.meta.ifix);
    iq = length(problem.meta.jlow) + length(problem.meta.ilow)+length(problem.meta.jupp) + length(problem.meta.iupp) + 2*(length(problem.meta.jrng) + length(problem.meta.irng))
    λ=x_a[problem.meta.nvar+1:problem.meta.nvar+iq+eq];
    ν=λ[eq+1:eq+iq];
    s=x_a[problem.meta.nvar+iq+eq+1:end];
    Jac = dresdx(x, problem);

    cons = res(x,problem);
    cons[eq+1:end] .= cons[eq+1:end] .+ s.^2
    return norm(grad(problem,x)+Jac'*λ)^2 + sum(cons.^2) + norm(2*ν.*s.^2 .-μ)^2
end

"""
  ipm_grad!(g, x_a, μ, problem::CUTEstModel)

Compute the gradient of the IPM objective function for a `CUTEstModel` problem.

# Arguments
- `g`: A preallocated vector to store the gradient result.
- `x_a`: A combined vector of decision variables (`x`), Lagrange multipliers (`λ`), and slack variables (`s`).
- `μ`: The barrier penalty parameter (scalar).
- `problem`: A `CUTEstModel` optimization problem.

# Notes
- The gradient is computed in place and stored in `g`.
"""
function ipm_grad!(g, x_a, μ, problem::CUTEstModel)

    x=x_a[1:problem.meta.nvar];
    jeq, jlo, jup, jrg = length(problem.meta.jfix), length(problem.meta.jlow), length(problem.meta.jupp), length(problem.meta.jrng);
    ieq, ilo, iup, irg = length(problem.meta.ifix), length(problem.meta.ilow), length(problem.meta.iupp), length(problem.meta.irng);
    eq=jeq+ieq;
    iq=jlo+ilo+jup+iup+2*jrg+2*irg;
    λ=x_a[problem.meta.nvar+1:problem.meta.nvar+iq+eq];
    ν=λ[eq+1:eq+iq];
    s=x_a[problem.meta.nvar+iq+eq+1:end];
    Jacobian = dresdx(x, problem);

    arr=zeros(jeq+jlo+jup+jrg);
    arr[problem.meta.jfix[1:jeq]]=λ[1:jeq]
    arr[problem.meta.jlow[1:jlo]]=-λ[jeq+ieq+1:jeq+ieq+jlo]
    arr[problem.meta.jupp[1:jup]]=λ[jeq+ieq+jlo+ilo+1:jeq+ieq+jlo+ilo+jup]
    arr[problem.meta.jrng[1:jrg]]=-λ[jeq+ieq+jlo+ilo+jup+iup+1:jeq+ieq+jlo+ilo+jup+iup+jrg] .+ λ[jeq+ieq+jlo+ilo+jup+iup+jrg+1:jeq+ieq+jlo+ilo+jup+iup+2*jrg]


    Hx = hess(problem, x, arr);
    L = grad(problem,x)+Jacobian'*λ;

    cons = res(x,problem);
    cons[eq+1:end] .= cons[eq+1:end] .+ s.^2;

    g[1:problem.meta.nvar] = 2*transpose(Hx)*(L) + 2*Jacobian'*cons
    g[problem.meta.nvar+1:problem.meta.nvar+length(λ)] = 2*Jacobian*L
    g[problem.meta.nvar+eq+1:problem.meta.nvar+length(λ)] .= g[problem.meta.nvar+eq+1:problem.meta.nvar+length(λ)] .+ 4*(2*ν.*s.^2 .- μ).*(s.^2)
    g[problem.meta.nvar+length(λ)+1:problem.meta.nvar+length(λ)+length(s)] = 4*cons[eq+1:end].*s
    g[problem.meta.nvar+length(λ)+eq+1:problem.meta.nvar+length(λ)+length(s)] .= g[problem.meta.nvar+length(λ)+eq+1:problem.meta.nvar+length(λ)+length(s)] .+ 8*(2*ν.*s.^2 .- μ).*ν.*s

end