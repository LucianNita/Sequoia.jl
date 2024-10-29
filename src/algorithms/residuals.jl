function r0(x,problem::CUTEstModel)
    constraint_val = cons(problem, x)
    eq_penalty_term = 0.0;
    lineq_penalty_term = 0.0;
    uineq_penalty_term = 0.0;
    rineq_penalty_term = 0.0;

    var_fix = 0.0;
    var_low = 0.0;
    var_upp = 0.0;
    var_rng = 0.0;

    # Handle equality constraints (penalty applied if violated)
    for i in eachindex(problem.meta.jfix)
        eq_penalty_term += 0.5 * (constraint_val[problem.meta.jfix[i]]-problem.meta.lcon[problem.meta.jfix[i]])^2
    end

    # Handle inequality constraints (penalty applied if outside bounds)
    for i in eachindex(problem.meta.jlow)
        if constraint_val[problem.meta.jlow[i]] < problem.meta.lcon[problem.meta.jlow[i]]
            lineq_penalty_term += 0.5 * (problem.meta.lcon[problem.meta.jlow[i]]-constraint_val[problem.meta.jlow[i]])^2
        end
    end

    for i in eachindex(problem.meta.jupp)
        if constraint_val[problem.meta.jupp[i]] > problem.meta.ucon[problem.meta.jupp[i]]
            uineq_penalty_term += 0.5 * (constraint_val[problem.meta.jupp[i]]-problem.meta.ucon[problem.meta.jupp[i]])^2
        end
    end

    for i in eachindex(problem.meta.jrng)
        if constraint_val[problem.meta.jrng[i]] > problem.meta.ucon[problem.meta.jrng[i]]
            rineq_penalty_term += 0.5 * (constraint_val[problem.meta.jrng[i]]-problem.meta.ucon[problem.meta.jrng[i]])^2
        end
        if constraint_val[problem.meta.jrng[i]] < problem.meta.lcon[problem.meta.jrng[i]]
            rineq_penalty_term += 0.5 * (problem.meta.lcon[problem.meta.jrng[i]]-constraint_val[problem.meta.jrng[i]])^2
        end
    end

    for i in eachindex(problem.meta.ifix)
        var_fix += 0.5 * (x[problem.meta.ifix[i]]-problem.meta.lvar[i])^2
    end

    # Handle inequality constraints (penalty applied if outside bounds)
    for i in eachindex(problem.meta.ilow)
        if x[problem.meta.ilow[i]] < problem.meta.lvar[problem.meta.ilow[i]]
            var_low += 0.5 * (problem.meta.lvar[problem.meta.ilow[i]]-x[problem.meta.ilow[i]])^2
        end
    end

    for i in eachindex(problem.meta.iupp)
        if x[problem.meta.iupp[i]] > problem.meta.uvar[problem.meta.iupp[i]]
            var_upp += 0.5 * (x[problem.meta.iupp[i]]-problem.meta.uvar[problem.meta.iupp[i]])^2
        end
    end

    for i in eachindex(problem.meta.irng)
        if x[problem.meta.irng[i]] < problem.meta.lvar[problem.meta.irng[i]]
            var_rng += 0.5 * (problem.meta.lvar[problem.meta.irng[i]]-x[problem.meta.irng[i]])^2
        end
        if x[problem.meta.irng[i]] > problem.meta.uvar[problem.meta.irng[i]]
            var_rng += 0.5 * (x[problem.meta.irng[i]]-problem.meta.uvar[problem.meta.irng[i]])^2
        end
    end


    return eq_penalty_term + lineq_penalty_term + uineq_penalty_term + rineq_penalty_term + var_fix + var_low + var_upp + var_rng 
end

function r0_gradient!(grad_storage, x, problem::CUTEstModel)

    # Get constraint values and the Jacobian
    constraint_val = cons(problem, x)
    jacobian = jac(problem, x)  # This will be a sparse matrix

    # Initialize penalty gradients
    grad_eq_penalty = zeros(length(x))
    grad_ineq_penalty = zeros(length(x))
    grad_var_penalty = zeros(length(x))

    # Equality constraints
    for i in eachindex(problem.meta.jfix)
        idx = problem.meta.jfix[i]
        violation = constraint_val[idx] - problem.meta.lcon[idx]
        grad_eq_penalty += jacobian[idx, :] * violation
    end

    # Inequality constraints: lower bounds
    for i in eachindex(problem.meta.jlow)
        idx = problem.meta.jlow[i]
        if constraint_val[idx] < problem.meta.lcon[idx]
            violation = constraint_val[idx] - problem.meta.lcon[idx]
            grad_ineq_penalty += jacobian[idx, :] * violation
        end
    end

    # Inequality constraints: upper bounds
    for i in eachindex(problem.meta.jupp)
        idx = problem.meta.jupp[i]
        if constraint_val[idx] > problem.meta.ucon[idx]
            violation = constraint_val[idx] - problem.meta.ucon[idx]
            grad_ineq_penalty += jacobian[idx, :] * violation 
        end
    end

    # Inequality constraints: range constraints
    for i in eachindex(problem.meta.jrng)
        idx = problem.meta.jrng[i]
        if constraint_val[idx] > problem.meta.ucon[idx]
            violation = constraint_val[idx] - problem.meta.ucon[idx]
            grad_ineq_penalty += jacobian[idx, :] * violation 
        end
        if constraint_val[idx] < problem.meta.lcon[idx]
            violation = constraint_val[idx] - problem.meta.lcon[idx]
            grad_ineq_penalty += jacobian[idx, :] * violation 
        end
    end

    # Variable penalties: fixed, lower, upper, and range
    for i in eachindex(problem.meta.ifix)
        idx = problem.meta.ifix[i]
        violation = x[idx] - problem.meta.lvar[idx]
        grad_var_penalty[idx] += violation
    end

    for i in eachindex(problem.meta.ilow)
        idx = problem.meta.ilow[i]
        if x[idx] < problem.meta.lvar[idx]
            violation = x[idx] - problem.meta.lvar[idx]
            grad_var_penalty[idx] += violation
        end
    end

    for i in eachindex(problem.meta.iupp)
        idx = problem.meta.iupp[i]
        if x[idx] > problem.meta.uvar[idx]
            violation = x[idx] - problem.meta.uvar[idx]
            grad_var_penalty[idx] += violation
        end
    end

    for i in eachindex(problem.meta.irng)
        idx = problem.meta.irng[i]
        if x[idx] > problem.meta.uvar[idx]
            violation = x[idx] - problem.meta.uvar[idx]
            grad_var_penalty[idx] += violation
        end
        if x[idx] < problem.meta.lvar[idx]
            violation = x[idx] - problem.meta.lvar[idx]
            grad_var_penalty[idx] += violation
        end
    end

    # Combine all gradients: objective + penalty terms
    grad_storage .= grad_eq_penalty .+ grad_ineq_penalty .+ grad_var_penalty

end

function r(x,tk,problem::CUTEstModel)
    cviol = r0(x,problem);
    obj_val = 0.0;

    obj_penalty = obj(problem, x)-tk;
    if obj_penalty>0.0
        obj_val=0.5*(obj_penalty)^2
    end

    return cviol+obj_val;
end

function r_gradient!(grad_storage, x, tk, problem::CUTEstModel)

    grad_obj = zeros(length(x))

    # Gradient of the objective function
    obj_viol=obj(problem,x) - tk
    if obj_viol > 0.0
        grad_obj = grad(problem, x) * (obj_viol)
    end

    r0_gradient!(grad_storage,x,problem);

    grad_storage .= grad_storage .+ grad_obj;

end


function exact_constraint_violation(x,problem::CUTEstModel)
    constraint_val = cons(problem, x)
    eq_violation = norm(constraint_val[problem.meta.jfix]-problem.meta.lcon[problem.meta.jfix])  # Equality violation
    lineq_violation = norm(max.(0.0, problem.meta.lcon[problem.meta.jlow]-constraint_val[problem.meta.jlow]))  # Inequality violation
    uineq_violation = norm(max.(0.0, constraint_val[problem.meta.jupp]-problem.meta.ucon[problem.meta.jupp]))  # Inequality violation
    rineq_violation = norm(max.(0.0, problem.meta.lcon[problem.meta.jrng]-constraint_val[problem.meta.jrng], constraint_val[problem.meta.jrng]-problem.meta.ucon[problem.meta.jrng]))

    var_fix_violation = norm(x[problem.meta.ifix]-problem.meta.lvar[problem.meta.ifix]);
    var_low_violation = norm(max.(0.0, problem.meta.lvar[problem.meta.ilow]-x[problem.meta.ilow]))  # Inequality violation
    var_upp_violation = norm(max.(0.0, x[problem.meta.iupp]-problem.meta.uvar[problem.meta.iupp]))
    var_rng_violation = norm(max.(0.0, problem.meta.lvar[problem.meta.irng]-x[problem.meta.irng], x[problem.meta.irng]-problem.meta.uvar[problem.meta.irng]))

    return eq_violation + lineq_violation + uineq_violation + rineq_violation + var_fix_violation + var_low_violation + var_upp_violation + var_rng_violation
end

function exact_augmented_constraint_violation(x,tk,problem::CUTEstModel)
    obj_violation = max(0.0, obj(problem,x)-tk)

    return obj_violation + exact_constraint_violation(x,problem)
end

function qpm_obj(x,μ,problem::CUTEstModel)
    return obj(problem,x)+μ*r0(x,problem)
end

function qpm_grad!(g, x, μ, problem::CUTEstModel)
    grad_obj = grad(problem,x);
    r0_gradient!(g,x,problem);
    g .=  grad_obj .+ μ .* g;
end



function r0(x,problem::SEQUOIA_pb)
    constraint_val = problem.constraints(x)
    eq_penalty_term = 0.0;
    ineq_penalty_term = 0.0;

    # Handle equality constraints (penalty applied if violated)
    for i in problem.eqcon
        eq_penalty_term += 0.5 * (constraint_val[i])^2
    end

    # Handle inequality constraints (penalty applied if outside bounds)
    for i in problem.ineqcon
        if constraint_val[i] > 0.0
            ineq_penalty_term += 0.5 * (constraint_val[i])^2 #can be replaced with max(0,fn)
        end
    end


    return eq_penalty_term + ineq_penalty_term
end
    # Helper function: In-place gradient of the augmented objective with penalty
    function r0_gradient!(g, x, problem::SEQUOIA_pb)
        constraint_val = problem.constraints(x)
        
        # Initialize penalty gradients
        grad_eq_penalty = zeros(length(x))
        grad_ineq_penalty = zeros(length(x))

        # Handle equality constraints
        for i in problem.eqcon
            grad_eq_penalty += problem.jacobian(x)[i, :] * (constraint_val[i])
        end

        # Handle inequality constraints
        for i in problem.ineqcon
            if constraint_val[i] > 0.0
                grad_ineq_penalty += problem.jacobian(x)[i, :] * (constraint_val[i])
            end
        end

        # Update the gradient storage with the objective gradient + penalties
        g .= grad_eq_penalty .+ grad_ineq_penalty
    end


function qpm_obj(x,μ,problem::SEQUOIA_pb)
    return problem.objective(x)+μ*r0(x,problem)
end

function qpm_grad!(g, x, μ, problem::SEQUOIA_pb)
    grad_obj = problem.gradient(x);
    r0_gradient!(g,x,problem);
    g .=  grad_obj .+ μ .* g;
end

function exact_constraint_violation(x,problem::SEQUOIA_pb)
    constraint_val = problem.constraints(x)
    eq_violation = norm(constraint_val[problem.eqcon])  # Equality violation
    ineq_violation = norm(max.(0.0, constraint_val[problem.ineqcon]))  # Inequality violation
    return eq_violation + ineq_violation
end

function r(x,tk,problem::SEQUOIA_pb)
    cviol = r0(x,problem);
    obj_val = 0.0;

    obj_penalty = problem.objective(x)-tk;
    if obj_penalty>0.0
        obj_val=0.5*(obj_penalty)^2
    end

    return cviol+obj_val;
end


function r_gradient!(grad_storage, x, tk, problem::SEQUOIA_pb)

    grad_obj = zeros(length(x))

    # Gradient of the objective function
    obj_viol=problem.objective(x) - tk
    if obj_viol > 0.0
        grad_obj = problem.gradient(x) * (obj_viol)
    end

    r0_gradient!(grad_storage,x,problem);

    grad_storage .= grad_storage .+ grad_obj;

end

function exact_augmented_constraint_violation(x,tk,problem::SEQUOIA_pb)
    obj_violation = max(0.0, problem.objective(x)-tk)

    return obj_violation + exact_constraint_violation(x,problem)
end

function auglag_obj(x, μ, λ, problem::SEQUOIA_pb)
    constraint_val = problem.constraints(x)
    for i in problem.ineqcon
        if constraint_val[i] <= 0.0
            constraint_val[i] = 0.0;
        end
    end
    return problem.objective(x) + μ*r0(x,problem) + constraint_val' * λ
end

function auglag_grad!(g, x, μ, λ, problem::SEQUOIA_pb)
    grad_obj = problem.gradient(x);
    jac=problem.jacobian(x);
    constraint_val = problem.constraints(x);
    for i in problem.ineqcon
        if constraint_val[i] <= 0.0
            jac[i,:] = zeros(length(x));
        end
    end
    r0_gradient!(g,x,problem);
    g .= grad_obj .+ μ .* g .+ jac' * λ #can be taylored
end


function update_lag_mult!(x, μ, λ, problem::SEQUOIA_pb)
    constraint_val = problem.constraints(x)
    for i in problem.eqcon
        λ[i] += μ * (constraint_val[i])
    end
    for i in problem.ineqcon
        if constraint_val[i] > 0.0
            λ[i] += μ * (constraint_val[i])
        end
    end
end