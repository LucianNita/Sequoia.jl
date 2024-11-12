export ipm_obj, ipm_grad!

"""
    ipm_obj(x_a, μ, problem::SEQUOIA_pb)

Compute the Interior Point Method (IPM) objective function for a `SEQUOIA_pb` problem.

# Arguments
- `x_a`: A combined vector of:
    - Decision variables (`x`).
    - Lagrange multipliers (`λ`) for constraints.
- `μ`: Barrier penalty parameter (scalar).
- `problem`: A `SEQUOIA_pb` optimization problem.

# Returns
- A scalar value representing the IPM objective:
  - The objective is the sum of:
      - Squared gradient norm of the Lagrangian.
      - Squared residuals of the constraints.
      - Barrier terms to enforce complementarity.
      - Squared penalty for dual feasibility violation. 

# Notes
- Handles both equality and inequality constraints.
"""
function ipm_obj(x_a, μ, problem::SEQUOIA_pb)
    
  x=x_a[1:problem.nvar];
  iq=length(problem.ineqcon);
  eq=length(problem.eqcon);
  λ=x_a[problem.nvar+1:problem.nvar+iq+eq];
  ν=λ[problem.ineqcon];
  jac=problem.jacobian(x);

  constraint_val = problem.constraints(x)
  cs=sum((ν.*constraint_val[problem.ineqcon] .-μ).^2);
  constraint_val[problem.ineqcon] .= max.( 0.0, constraint_val[problem.ineqcon]) 
  return sum( (problem.gradient(x)+jac'*λ).^2 )+ sum(constraint_val.^2) + cs + sum((max.(0.0, -ν) ).^2 )
end

"""
    ipm_grad!(g, x_a, μ, problem::SEQUOIA_pb)

Compute the gradient of the IPM objective function for a `SEQUOIA_pb` problem.

# Arguments
- `g`: A preallocated vector to store the gradient.
- `x_a`: A combined vector of:
    - Decision variables (`x`).
    - Lagrange multipliers (`λ`) for constraints.
- `μ`: Barrier penalty parameter (scalar).
- `problem`: A `SEQUOIA_pb` optimization problem.

# Notes
- Computes the gradient using automatic differentiation via `ForwardDiff`.
"""
function ipm_grad!(g, x_a, μ, problem::SEQUOIA_pb)
  ForwardDiff.gradient!(g, z -> ipm_obj(z, μ, problem), x_a)
end 

"""
    ipm_obj(x_a, μ, problem::CUTEstModel)

Compute the Interior Point Method (IPM) objective function for a `CUTEstModel` problem.

# Arguments
- `x_a`: A combined vector of:
    - Decision variables (`x`).
    - Lagrange multipliers (`λ`) for constraints.
- `μ`: Barrier penalty parameter (scalar).
- `problem`: A `CUTEstModel` optimization problem.

# Returns
- A scalar value representing the IPM objective.

# Notes
- Incorporates both equality and inequality constraints from the `CUTEstModel`.
"""
function ipm_obj(x_a, μ, problem::CUTEstModel)
  x=x_a[1:problem.meta.nvar];
  eq = length(problem.meta.jfix) + length(problem.meta.ifix);
  iq = length(problem.meta.jlow) + length(problem.meta.ilow)+length(problem.meta.jupp) + length(problem.meta.iupp) + 2*(length(problem.meta.jrng) + length(problem.meta.irng))
  λ=x_a[problem.meta.nvar+1:problem.meta.nvar+iq+eq];
  ν=λ[eq+1:eq+iq];
  Jac = dresdx(x, problem);

  cons = res(x,problem);
  cs=sum((ν.*cons[eq+1:end] .-μ).^2);
  cons[eq+1:end] .= max.(0.0, cons[eq+1:end])
  return norm(grad(problem,x)+Jac'*λ)^2 + sum(cons.^2) + cs + sum((max.(0.0, -ν) ).^2 )
end

"""
  ipm_grad!(g, x_a, μ, problem::CUTEstModel)

Compute the gradient of the IPM objective function for a `CUTEstModel` problem.

# Arguments
- `g`: A preallocated vector to store the gradient result.
- `x_a`: A combined vector of decision variables (`x`), Lagrange multipliers (`λ`).
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
  Jacobian = dresdx(x, problem);

  arr=zeros(jeq+jlo+jup+jrg);
  arr[problem.meta.jfix[1:jeq]]=λ[1:jeq]
  arr[problem.meta.jlow[1:jlo]]=-λ[jeq+ieq+1:jeq+ieq+jlo]
  arr[problem.meta.jupp[1:jup]]=λ[jeq+ieq+jlo+ilo+1:jeq+ieq+jlo+ilo+jup]
  arr[problem.meta.jrng[1:jrg]]=-λ[jeq+ieq+jlo+ilo+jup+iup+1:jeq+ieq+jlo+ilo+jup+iup+jrg] .+ λ[jeq+ieq+jlo+ilo+jup+iup+jrg+1:jeq+ieq+jlo+ilo+jup+iup+2*jrg]


  Hx = hess(problem, x, arr);
  L = grad(problem,x)+Jacobian'*λ;

  cons = res(x,problem);
  cold = cons[eq+1:end];
  cons[eq+1:end] .= max.(0.0, cons[eq+1:end]);

  g[1:problem.meta.nvar] = 2*transpose(Hx)*(L) + 2*Jacobian'*cons + 2*(Jacobian[eq+1:end,:])'*(ν.*(ν.*cold .-μ))
  g[problem.meta.nvar+1:problem.meta.nvar+length(λ)] = 2*Jacobian*L 
  g[problem.meta.nvar+eq+1:problem.meta.nvar+length(λ)] .= g[problem.meta.nvar+eq+1:problem.meta.nvar+length(λ)] .+ 2*cold.*(ν.*cold .-μ) .- 2.0*max.(0.0, -ν) 

end