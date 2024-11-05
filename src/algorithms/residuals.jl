export r0, r0_gradient!, r, r_gradient!, exact_constraint_violation, exact_augmented_constraint_violation,
       qpm_obj, qpm_grad!, r0_gradient!, auglag_obj, auglag_grad!, update_lag_mult!,
       ipm_obj, ipm_grad!, update_ipm_mult!
       
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
        λ[i] = max(0.0, λ[i] + μ * (constraint_val[i]))
        #if constraint_val[i] > 0.0
        #    λ[i] += μ * (constraint_val[i])
        #end
    end
end

function auglag_obj(x, μ, λ, problem::CUTEstModel)
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
        c=(constraint_val[problem.meta.jfix[i]]-problem.meta.lcon[problem.meta.jfix[i]]);
        eq_penalty_term += 0.5 * μ * c^2 + λ[i] * c
    end
    past = length(problem.meta.jfix);
    # Handle inequality constraints (penalty applied if outside bounds)
    for i in eachindex(problem.meta.jlow)
        if constraint_val[problem.meta.jlow[i]] < problem.meta.lcon[problem.meta.jlow[i]]
            c=(problem.meta.lcon[problem.meta.jlow[i]]-constraint_val[problem.meta.jlow[i]]);
            lineq_penalty_term += 0.5 * μ * c^2 + λ[i+past] * c
        end
    end
    past+= length(problem.meta.jlow)

    for i in eachindex(problem.meta.jupp)
        if constraint_val[problem.meta.jupp[i]] > problem.meta.ucon[problem.meta.jupp[i]]
            c=(constraint_val[problem.meta.jupp[i]]-problem.meta.ucon[problem.meta.jupp[i]]);
            uineq_penalty_term += 0.5 * μ * c^2 + λ[i+past] * c
        end
    end
    past+= length(problem.meta.jupp)

    for i in eachindex(problem.meta.jrng)
        if constraint_val[problem.meta.jrng[i]] > problem.meta.ucon[problem.meta.jrng[i]]
            c=(constraint_val[problem.meta.jrng[i]]-problem.meta.ucon[problem.meta.jrng[i]]);
            rineq_penalty_term += 0.5 * μ * c^2 + λ[i+past] * c
        end
        if constraint_val[problem.meta.jrng[i]] < problem.meta.lcon[problem.meta.jrng[i]]
            c=(problem.meta.lcon[problem.meta.jrng[i]]-constraint_val[problem.meta.jrng[i]]);
            rineq_penalty_term += 0.5 * μ * c^2 + λ[i+past] * c
        end
    end
    past+=length(problem.meta.jrng)

    for i in eachindex(problem.meta.ifix)
        c=(x[problem.meta.ifix[i]]-problem.meta.lvar[i])
        var_fix += 0.5 * μ * c^2 + λ[i+past] * c
    end
    past+=length(problem.meta.ifix)

    # Handle inequality constraints (penalty applied if outside bounds)
    for i in eachindex(problem.meta.ilow)
        if x[problem.meta.ilow[i]] < problem.meta.lvar[problem.meta.ilow[i]]
            c=(problem.meta.lvar[problem.meta.ilow[i]]-x[problem.meta.ilow[i]])
            var_low += 0.5 * μ * c^2 + λ[i+past] * c
        end
    end
    past+=length(problem.meta.ilow)

    for i in eachindex(problem.meta.iupp)
        if x[problem.meta.iupp[i]] > problem.meta.uvar[problem.meta.iupp[i]]
            c=(x[problem.meta.iupp[i]]-problem.meta.uvar[problem.meta.iupp[i]])
            var_upp += 0.5 * μ * c^2 + λ[i+past] * c
        end
    end
    past+=length(problem.meta.iupp)

    for i in eachindex(problem.meta.irng)
        if x[problem.meta.irng[i]] < problem.meta.lvar[problem.meta.irng[i]]
            c=(problem.meta.lvar[problem.meta.irng[i]]-x[problem.meta.irng[i]])
            var_rng += 0.5 * μ * c^2 + λ[i+past] * c
        end
        if x[problem.meta.irng[i]] > problem.meta.uvar[problem.meta.irng[i]]
            c=(x[problem.meta.irng[i]]-problem.meta.uvar[problem.meta.irng[i]])
            var_rng += 0.5 * μ * c^2 + λ[i+past] * c
        end
    end

    return obj(problem, x) + eq_penalty_term + lineq_penalty_term + uineq_penalty_term + rineq_penalty_term + var_fix + var_low + var_upp + var_rng 
end

function auglag_grad!(g, x, μ, λ, problem::CUTEstModel)
    grad_obj = grad(problem, x);
    jacobian = jac(problem, x);
    constraint_val = cons(problem, x);

    # Initialize penalty gradients
    grad_eq_penalty = zeros(length(x))
    grad_ineq_penalty = zeros(length(x))
    grad_var_penalty = zeros(length(x))

    # Equality constraints
    for i in eachindex(problem.meta.jfix)
        idx = problem.meta.jfix[i]
        violation = constraint_val[idx] - problem.meta.lcon[idx]
        grad_eq_penalty += jacobian[idx, :] * (μ * violation + λ[i])
    end
    past = length(problem.meta.jfix);
        # Inequality constraints: lower bounds
    for i in eachindex(problem.meta.jlow)
        idx = problem.meta.jlow[i]
        if constraint_val[idx] < problem.meta.lcon[idx]
            violation = constraint_val[idx] - problem.meta.lcon[idx]
            grad_ineq_penalty += jacobian[idx, :] * (μ * violation + λ[i+past])
        end
    end
    past += length(problem.meta.jlow);
    # Inequality constraints: upper bounds
    for i in eachindex(problem.meta.jupp)
        idx = problem.meta.jupp[i]
        if constraint_val[idx] > problem.meta.ucon[idx]
            violation = constraint_val[idx] - problem.meta.ucon[idx]
            grad_ineq_penalty += jacobian[idx, :] *  (μ * violation + λ[i+past])
        end
    end
    past += length(problem.meta.jupp);
    # Inequality constraints: range constraints
    for i in eachindex(problem.meta.jrng)
        idx = problem.meta.jrng[i]
        if constraint_val[idx] > problem.meta.ucon[idx]
            violation = constraint_val[idx] - problem.meta.ucon[idx]
            grad_ineq_penalty += jacobian[idx, :] * (μ * violation + λ[i+past])
        end
        if constraint_val[idx] < problem.meta.lcon[idx]
            violation = constraint_val[idx] - problem.meta.lcon[idx]
            grad_ineq_penalty += jacobian[idx, :] * (μ * violation + λ[i+past])
        end
    end
    past += length(problem.meta.jrng);
    # Variable penalties: fixed, lower, upper, and range
    for i in eachindex(problem.meta.ifix)
        idx = problem.meta.ifix[i]
        violation = x[idx] - problem.meta.lvar[idx]
        grad_var_penalty[idx] += (μ * violation + λ[i+past])
    end

    past+=length(problem.meta.ifix);
    for i in eachindex(problem.meta.ilow)
        idx = problem.meta.ilow[i]
        if x[idx] < problem.meta.lvar[idx]
            violation = x[idx] - problem.meta.lvar[idx]
            grad_var_penalty[idx] += (μ * violation + λ[i+past])
        end
    end

    past+=length(problem.meta.ilow);
    for i in eachindex(problem.meta.iupp)
        idx = problem.meta.iupp[i]
        if x[idx] > problem.meta.uvar[idx]
            violation = x[idx] - problem.meta.uvar[idx]
            grad_var_penalty[idx] += (μ * violation + λ[i+past])
        end
    end

    past+=length(problem.meta.iupp);
    for i in eachindex(problem.meta.irng)
        idx = problem.meta.irng[i]
        if x[idx] > problem.meta.uvar[idx]
            violation = x[idx] - problem.meta.uvar[idx]
            grad_var_penalty[idx] += (μ * violation + λ[i+past])
        end
        if x[idx] < problem.meta.lvar[idx]
            violation = x[idx] - problem.meta.lvar[idx]
            grad_var_penalty[idx] += (μ * violation + λ[i+past])
        end
    end

    g .= grad_obj .+ grad_eq_penalty .+ grad_ineq_penalty .+ grad_var_penalty
end


function update_lag_mult!(x, μ, λ, problem::CUTEstModel)
    constraint_val = cons(problem, x)
    
    # Handle equality constraints (penalty applied if violated)
    for i in eachindex(problem.meta.jfix)
        c=(constraint_val[problem.meta.jfix[i]]-problem.meta.lcon[problem.meta.jfix[i]]);
        λ[i] += μ * c
    end
    past = length(problem.meta.jfix);

    # Handle inequality constraints (penalty applied if outside bounds)
    for i in eachindex(problem.meta.jlow)
        if constraint_val[problem.meta.jlow[i]] < problem.meta.lcon[problem.meta.jlow[i]]
            c=(problem.meta.lcon[problem.meta.jlow[i]]-constraint_val[problem.meta.jlow[i]]);
            λ[i+past] = max(0.0, λ[i+past] + μ * c)
        end
    end
    past+= length(problem.meta.jlow)

    for i in eachindex(problem.meta.jupp)
        if constraint_val[problem.meta.jupp[i]] > problem.meta.ucon[problem.meta.jupp[i]]
            c=(constraint_val[problem.meta.jupp[i]]-problem.meta.ucon[problem.meta.jupp[i]]);
            λ[i+past] = max(0.0, λ[i+past] + μ * c)
        end
    end
    past+= length(problem.meta.jupp)

    for i in eachindex(problem.meta.jrng)
        if constraint_val[problem.meta.jrng[i]] > problem.meta.ucon[problem.meta.jrng[i]]
            c=(constraint_val[problem.meta.jrng[i]]-problem.meta.ucon[problem.meta.jrng[i]]);
            λ[i+past] = max(0.0, λ[i+past] + μ * c)
        end
        if constraint_val[problem.meta.jrng[i]] < problem.meta.lcon[problem.meta.jrng[i]]
            c=(problem.meta.lcon[problem.meta.jrng[i]]-constraint_val[problem.meta.jrng[i]]);
            λ[i+past] = max(0.0, λ[i+past] + μ * c)
        end
    end
    past+=length(problem.meta.jrng)

    for i in eachindex(problem.meta.ifix)
        c=(x[problem.meta.ifix[i]]-problem.meta.lvar[i])
        λ[i+past] += (μ * c)
    end
    past+=length(problem.meta.ifix)

    # Handle inequality constraints (penalty applied if outside bounds)
    for i in eachindex(problem.meta.ilow)
        if x[problem.meta.ilow[i]] < problem.meta.lvar[problem.meta.ilow[i]]
            c=(problem.meta.lvar[problem.meta.ilow[i]]-x[problem.meta.ilow[i]])
            λ[i+past] = max(0.0, λ[i+past] + μ * c)
        end
    end
    past+=length(problem.meta.ilow)

    for i in eachindex(problem.meta.iupp)
        if x[problem.meta.iupp[i]] > problem.meta.uvar[problem.meta.iupp[i]]
            c=(x[problem.meta.iupp[i]]-problem.meta.uvar[problem.meta.iupp[i]])
            λ[i+past] = max(0.0, λ[i+past] + μ * c)
        end
    end
    past+=length(problem.meta.iupp)

    for i in eachindex(problem.meta.irng)
        if x[problem.meta.irng[i]] < problem.meta.lvar[problem.meta.irng[i]]
            c=(problem.meta.lvar[problem.meta.irng[i]]-x[problem.meta.irng[i]])
            λ[i+past] = max(0.0, λ[i+past] + μ * c)
        end
        if x[problem.meta.irng[i]] > problem.meta.uvar[problem.meta.irng[i]]
            c=(x[problem.meta.irng[i]]-problem.meta.uvar[problem.meta.irng[i]])
            λ[i+past] = max(0.0, λ[i+past] + μ * c)
        end
    end    

end

function ipm_obj(x_a, μ, problem::SEQUOIA_pb)
    x=x_a[1:problem.nvar];
    iq=length(problem.ineqcon);
    eq=length(problem.eqcon);
    λ=x_a[problem.nvar+1:problem.nvar+iq+eq];
    s=x_a[problem.nvar+iq+eq+1:end];
    jac=problem.jacobian(x);
    #display(s)
    ν=λ[problem.ineqcon];
    return norm(problem.gradient(x)+jac'*λ)^2+ r_slack(x,s,problem) + norm(2*ν.*s.^2 .-μ)^2 #+ sum(max(-λ[i], 0)^2 for i in 1:problem.ineqcon) + sum(λ[problem.ineqcon[i]] * (problem.constraints(x)[problem.ineqcon[i]]+s[i]^2)^2 for i in 1:iq)
end

function r_slack(x,s,problem::SEQUOIA_pb)
    constraint_val = problem.constraints(x)
    eq_penalty_term = 0.0;
    ineq_penalty_term = 0.0;

    # Handle equality constraints (penalty applied if violated)
    for i in problem.eqcon
        eq_penalty_term += (constraint_val[i])^2
    end

    # Handle inequality constraints (penalty applied if outside bounds)
    for i in eachindex(problem.ineqcon)
        ineq_penalty_term += (constraint_val[problem.ineqcon[i]]+s[i]^2)^2 
    end

    return eq_penalty_term + ineq_penalty_term
end

function ipm_grad!(g, x_a, μ, problem::SEQUOIA_pb)

    #x=x_a[1:problem.nvar];
    #λ=x_a[problem.nvar+1:problem.nvar+problem.ineqcon+problem.eqcon];
    #s=x_a[problem.nvar+problem.ineqcon+problem.eqcon+1:end];

    g[1:end] .= 0.0;

    #grad_obj=problem.gradient
    #jacobian=problem.jacobian(x);
    #constraint_val = problem.constraints(x);
    #ν = λ[problem.ineqcon];

    g .= ForwardDiff.gradient(z -> ipm_obj(z, μ, problem), x_a)

end

function res(x,problem::CUTEstModel)
    
    jeq=length(problem.meta.jfix);
    jlo=length(problem.meta.jlow);
    jup=length(problem.meta.jupp);
    jrg=length(problem.meta.jrng);
    ieq=length(problem.meta.ifix);
    ilo=length(problem.meta.ilow);
    iup=length(problem.meta.iupp);
    irg=length(problem.meta.irng);

    violation=zeros(jeq+jlo+jup+2*jrg+ieq+ilo+iup+2*irg);
    constraint_val = cons(problem, x)
    
    # Handle equality constraints (penalty applied if violated)
    for i in 1:jeq
        violation[i] = constraint_val[problem.meta.jfix[i]]-problem.meta.lcon[problem.meta.jfix[i]]
    end
    for i in 1:ieq
        violation[jeq+i] = x[problem.meta.ifix[i]]-problem.meta.lvar[i]
    end

    # Handle inequality constraints (penalty applied if outside bounds)
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
        violation[jeq+ieq+jlo+ilo+jup+iup+2*jrg+jig+i] = x[problem.meta.irng[i]]-problem.meta.uvar[problem.meta.irng[i]]
    end


    return violation
end

function r_slack(x,s,problem::CUTEstModel)
    cons=res(x,problem);
    sum=0.0;

    jeq=length(problem.meta.jfix);
    jlo=length(problem.meta.jlow);
    jup=length(problem.meta.jupp);
    jrg=length(problem.meta.jrng);
    ieq=length(problem.meta.ifix);
    ilo=length(problem.meta.ilow);
    iup=length(problem.meta.iupp);
    irg=length(problem.meta.irng);

    for i in 1:jeq+ieq
        sum +=cons[i]^2;
    end
    for i in 1:jlo+ilo+jup+iup+2*jrg+2*irg
        sum +=(cons[i+jeq+ieq]+s[i]^2)^2
    end
    return sum
end

function ipm_obj(x_a, μ, problem::CUTEstModel)
    x=x_a[1:problem.nvar];
    jeq=length(problem.meta.jfix);
    jlo=length(problem.meta.jlow);
    jup=length(problem.meta.jupp);
    jrg=length(problem.meta.jrng);
    ieq=length(problem.meta.ifix);
    ilo=length(problem.meta.ilow);
    iup=length(problem.meta.iupp);
    irg=length(problem.meta.irng);

    eq=jeq+ieq;
    iq=jlo+ilo+jup+iup+2*jrg+2*irg;

    λ=x_a[problem.nvar+1:problem.nvar+iq+eq];
    s=x_a[problem.nvar+iq+eq+1:end];
    jac=problem.jacobian(x);
    ν=λ[eq+1:eq+iq];

    return norm(problem.gradient(x)+jac'*λ)^2 + r_slack(x,s,problem) + norm(2*ν.*s.^2 .-μ)^2
end

function ipm_grad!(g, x_a, μ, problem::CUTEstModel)

    x=x_a[1:problem.nvar];
    jeq=length(problem.meta.jfix);
    jlo=length(problem.meta.jlow);
    jup=length(problem.meta.jupp);
    jrg=length(problem.meta.jrng);
    ieq=length(problem.meta.ifix);
    ilo=length(problem.meta.ilow);
    iup=length(problem.meta.iupp);
    irg=length(problem.meta.irng);

    eq=jeq+ieq;
    iq=jlo+ilo+jup+iup+2*jrg+2*irg;

    λ=x_a[problem.nvar+1:problem.nvar+iq+eq];
    s=x_a[problem.nvar+iq+eq+1:end];

    Hx = hess(problem, x; y=)
end
