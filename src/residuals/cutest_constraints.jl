export res, dresdx

"""
    res(x, problem::CUTEstModel)

Compute the violation of equality, inequality, and range constraints for a CUTEst problem.

# Arguments
- `x`: The vector of variables.
- `problem::CUTEstModel`: A CUTEst optimization problem.

# Returns
- A vector of violations, structured as follows:
    1. Equality constraint violations (jeq): The first `jeq` elements correspond to the violation of equality constraints, `c_i(x) - lcon[i] = 0`, where `c_i` are the equality constraints defined in `jfix`.
    2. Variable equality violations (ieq): The next `ieq` elements correspond to variable equality constraints, `x[i] - lvar[i] = 0`, where indices are defined in `ifix`.
    3. Lower bound constraint violations (jlo): The next `jlo` elements represent lower-bound constraint violations, `lcon[i] - c_i(x) >= 0`, where indices are defined in `jlow`.
    4. Lower bound variable violations (ilo): The next `ilo` elements correspond to lower-bound variable violations, `lvar[i] - x[i] >= 0`, where indices are defined in `ilow`.
    5. Upper bound constraint violations (jup): The next `jup` elements represent upper-bound constraint violations, `c_i(x) - ucon[i] >= 0`, where indices are defined in `jupp`.
    6. Upper bound variable violations (iup): The next `iup` elements correspond to upper-bound variable violations, `x[i] - uvar[i] >= 0`, where indices are defined in `iupp`.
    7. Constraint range violations (2 * jrg): For each `jrg`, two entries correspond to:
        - Lower range violation: `lcon[i] - c_i(x)`.
        - Upper range violation: `c_i(x) - ucon[i]`.
    8. Variable range violations (2 * irg): For each `irg`, two entries correspond to:
        - Lower box violation: `lvar[i] - x[i]`.
        - Upper box violation: `x[i] - uvar[i]`.
"""
function res(x, problem::CUTEstModel)
    # Extract metadata
    meta = problem.meta
    jeq, jlo, jup, jrg = length(meta.jfix), length(meta.jlow), length(meta.jupp), length(meta.jrng)
    ieq, ilo, iup, irg = length(meta.ifix), length(meta.ilow), length(meta.iupp), length(meta.irng)

    # Initialize violation vector
    total_size = jeq + jlo + jup + 2*jrg + ieq + ilo + iup + 2*irg
    violation = Vector{Float64}(undef, total_size)
    constraint_val = cons(problem, x)
    
    # Handle equality constraints 
    for i in 1:jeq
        violation[i] = constraint_val[problem.meta.jfix[i]]-problem.meta.lcon[problem.meta.jfix[i]]
    end
    for i in 1:ieq
        violation[jeq+i] = x[problem.meta.ifix[i]]-problem.meta.lvar[problem.meta.ifix[i]]
    end

    # Handle inequality constraints 
    for i in 1:jlo
        violation[jeq+ieq+i] = problem.meta.lcon[problem.meta.jlow[i]]-constraint_val[problem.meta.jlow[i]]
    end
    for i in 1:ilo
        violation[jeq+ieq+jlo+i] = problem.meta.lvar[problem.meta.ilow[i]]-x[problem.meta.ilow[i]]
    end

    for i in 1:jup
        violation[jeq+ieq+jlo+ilo+i] = constraint_val[problem.meta.jupp[i]]-problem.meta.ucon[problem.meta.jupp[i]]
    end
    for i in 1:iup
        violation[jeq+ieq+jlo+ilo+jup+i] = x[problem.meta.iupp[i]]-problem.meta.uvar[problem.meta.iupp[i]]
    end

    for i in 1:jrg
        violation[jeq+ieq+jlo+ilo+jup+iup+i] = problem.meta.lcon[problem.meta.jrng[i]]-constraint_val[problem.meta.jrng[i]]
        violation[jeq+ieq+jlo+ilo+jup+iup+jrg+i] = constraint_val[problem.meta.jrng[i]]-problem.meta.ucon[problem.meta.jrng[i]]   
    end
    for i in 1:irg
        violation[jeq+ieq+jlo+ilo+jup+iup+2*jrg+i] = problem.meta.lvar[problem.meta.irng[i]]-x[problem.meta.irng[i]]
        violation[jeq+ieq+jlo+ilo+jup+iup+2*jrg+irg+i] = x[problem.meta.irng[i]]-problem.meta.uvar[problem.meta.irng[i]]
    end


    return violation
end

"""
    dresdx(x, problem::CUTEstModel)

Compute the Jacobian of the constraint violation vector for a CUTEst problem.

# Arguments
- `x`: The vector of variables.
- `problem::CUTEstModel`: A CUTEst optimization problem.

# Returns
- A sparse Jacobian matrix, `dres`, of size `(total_constraints x nvar)`:
    - Rows correspond to the constraints returned by `res`.
    - Columns correspond to the partial derivatives of the constraints with respect to variables `x`.

# Notes:
- Equality constraints (`jeq` rows):
  - These rows correspond to the derivatives of equality constraints ( c_i(x) ) with respect to ( x ).
  - Each row represents the gradient of an equality constraint indexed in `meta.jfix`.
  - For a constraint ( c_i(x) - lcon[i] = 0 ), the derivative is ( ∇c_i(x) ).

- Variable equality constraints (`ieq` rows):
  - These rows represent equality conditions on variables, where ( x[i] = lvar[i] ) for indices in `meta.ifix`.
  - Each row is a sparse vector with a single nonzero entry of `1.0` at the index corresponding to the variable being constrained.

- Lower-bound inequality constraints (`jlo` rows):
  - These rows represent lower-bound constraints on the constraints ( c_i(x) ), where ( lcon[i] - c_i(x) ≥ 0 ).
  - The derivative for such constraints is the negation of ( ∇c_i(x) ), ensuring the violation vector reflects the direction of feasibility.

- Lower-bound variable constraints (`ilo` rows):
  - These rows represent lower-bound conditions on variables ( lvar[i] - x[i] ≥ 0 ).
  - Each row is a sparse vector with a single nonzero entry of `-1.0` at the index of the constrained variable.

- Upper-bound inequality constraints (`jup` rows):
  - These rows represent upper-bound constraints on the constraints ( c_i(x) ), where ( c_i(x) - ucon[i] ≥ 0 ).
  - The derivative for such constraints is ( ∇c_i(x) ).

- Upper-bound variable constraints (`iup` rows):
  - These rows represent upper-bound conditions on variables ( x[i] - uvar[i] ≥ 0 ).
  - Each row is a sparse vector with a single nonzero entry of `1.0` at the index of the constrained variable.

- Range constraints (`jrg` rows):
  - For constraints with both lower and upper bounds (( lcon[i] ≤ c_i(x) ≤ ucon[i] )), two rows are included:
    1. Lower range violation: ( lcon[i] - c_i(x) ) with derivative ( -∇c_i(x) ).
    2. Upper range violation: ( c_i(x) - ucon[i] ) with derivative ( ∇c_i(x) ).

- Range variable constraints (`irg` rows):
  - For variables with both lower and upper bounds (( lvar[i] ≤ x[i] ≤ uvar[i] )), two rows are included:
    1. Lower bound violation: ( lvar[i] - x[i] ), with derivative ( -1.0 ) for the constrained variable.
    2. Upper bound violation: ( x[i] - uvar[i] ), with derivative ( 1.0 ) for the constrained variable.
"""
function dresdx(x, problem::CUTEstModel)
    # Extract metadata
    meta = problem.meta
    jeq, jlo, jup, jrg = length(meta.jfix), length(meta.jlow), length(meta.jupp), length(meta.jrng)
    ieq, ilo, iup, irg = length(meta.ifix), length(meta.ilow), length(meta.iupp), length(meta.irng)

    J = jac(problem,x);

    # Initialize violation vector
    total_size = jeq + jlo + jup + 2*jrg + ieq + ilo + iup + 2*irg
    dres = spzeros(total_size, meta.nvar)
    
    # Handle equality constraints 
    for i in 1:jeq
        dres[i,:] = J[problem.meta.jfix[i],:]
    end
    for i in 1:ieq
        dres[i+jeq,problem.meta.ifix[i]] = 1.0;
    end

    # Handle inequality constraints 
    for i in 1:jlo
        dres[i+jeq+ieq,:] = -J[problem.meta.jlow[i],:] 
    end
    for i in 1:ilo
        dres[i+jeq+ieq+jlo,problem.meta.ilow[i]] = -1.0;
    end

    for i in 1:jup
        dres[i+jeq+ieq+jlo+ilo,:] = J[problem.meta.jupp[i],:]  
    end
    for i in 1:iup
        dres[i+jeq+ieq+jlo+ilo+jup,problem.meta.iupp[i]] = 1.0;
    end

    for i in 1:jrg
        dres[i+jeq+ieq+jlo+ilo+jup+iup,:] = -J[problem.meta.jrng[i],:] 
        dres[i+jeq+ieq+jlo+ilo+jup+iup+jrg,:] = J[problem.meta.jrng[i],:]  
    end
    for i in 1:irg
        dres[i+jeq+ieq+jlo+ilo+jup+iup+2*jrg,problem.meta.irng[i]] = -1.0;
        dres[i+jeq+ieq+jlo+ilo+jup+iup+2*jrg+irg,problem.meta.irng[i]] = 1.0;
    end

    return dres
end