export ipm_solve!

# Interior Point Method (IPM) implementation
function ipm_solve!(problem::SEQUOIA_pb, inner_solver, options, time, x, previous_fval, iteration)

    # Initialize variables
    penalty_init=problem.solver_settings.solver_params[1];
    penalty_mult=problem.solver_settings.solver_params[2];
    damping_factor=problem.solver_settings.solver_params[3];
    penalty_param = penalty_init  # Set initial penalty parameter
    λ = problem.solver_settings.solver_params[5:end];  # Initialize or warm-start Lagrange multipliers
    n_con=length(λ);
    if problem.cutest_nlp === nothing
        s = ones(length(problem.ineqcon))
    else
        s = ones(n_con-length(problem.cutest_nlp.meta.ifix)-length(problem.cutest_nlp.meta.jfix))
    end
    
    result=nothing;
    constraint_violation=nothing;

    while iteration < problem.solver_settings.max_iter_outer && time < problem.solver_settings.max_time_outer

        if problem.cutest_nlp === nothing
            obj_aug_fn = x -> ipm_obj(x, penalty_param, problem)
            grad_aug_fn! = (g, x) -> ipm_grad!(g, x, penalty_param, problem)
        else
            obj_aug_fn = x -> ipm_obj(x, penalty_param, problem.cutest_nlp)
            grad_aug_fn! = (g, x) -> ipm_grad!(g, x, penalty_param, problem.cutest_nlp)
        end

        # Solve the unconstrained subproblem
        result = Optim.optimize(obj_aug_fn, grad_aug_fn!, vcat(x,λ,s), inner_solver, options)
        # Extract the optimized solution from the subproblem
        x = result.minimizer[1:problem.nvar]
        λ = result.minimizer[problem.nvar+1:problem.nvar+n_con]
        s = result.minimizer[problem.nvar+n_con+1:end]

        if problem.cutest_nlp === nothing
            # Compute the constraint violation for adaptive updates
            constraint_violation = exact_constraint_violation(x,problem)
        else
            constraint_violation = exact_constraint_violation(x,problem.cutest_nlp)
        end

        # Save a SEQUOIA_Solution_step after each optimize call
        fval = problem.objective(x)  # Objective function value
        time+= result.time_run  # Computation time
        
        conv = constraint_violation < problem.solver_settings.resid_tolerance && Optim.converged(result) && abs(fval - previous_fval) < problem.solver_settings.cost_tolerance #/(max(1.0,abs(previous_fval)))
        if conv
            if problem.solver_settings.store_trace
                x_tr=Optim.x_trace(result);
            else
                x_tr=nothing;
            end
            step = SEQUOIA_Solution_step(iteration, abs(fval - previous_fval), :first_order, result.time_run, result.iterations, x, fval, problem.gradient(x), problem.constraints(x),  vcat(penalty_param,λ,s), x_tr)
            add_iterate!(problem.solution_history, step)  # Add step to history

            previous_fval=fval;

            return time, x, previous_fval, iteration
        end

        if problem.solver_settings.store_trace
            step = SEQUOIA_Solution_step(iteration, abs(fval - previous_fval), :unknown, result.time_run, result.iterations, x, fval, problem.gradient(x), problem.constraints(x), vcat(penalty_param,λ,s),Optim.x_trace(result) )
            add_iterate!(problem.solution_history, step)  # Add step to history
        end

        # Update previous function value for the next iteration
        previous_fval = fval
      

        if problem.solver_settings.solver_params[4]==0
            penalty_param *= penalty_mult;
        else
            penalty_param *= max(min(1, problem.solver_settings.resid_tolerance / constraint_violation), damping_factor);
        end 
        
        # Increment the iteration counter & time
        iteration += 1
    end
    
    if iteration >= problem.solver_settings.max_iter_outer
        solver_status = :max_iter
    elseif time >= problem.solver_settings.max_time_outer
        solver_status = :max_time
    else
        solver_status = :unknown
    end
    if problem.solver_settings.store_trace
        problem.solution_history.iterates[end].solver_status=solver_status;
    else
        step = SEQUOIA_Solution_step(iteration, abs(result.minimum - previous_fval), solver_status, result.time_run, result.iterations, result.minimizer[1:problem.nvar], result.minimum, problem.gradient(x), problem.constraints(result.minimizer), vcat(penalty_param,λ,s))
        add_iterate!(problem.solution_history, step)  # Add step to history
    end    

    return time, x, previous_fval, iteration
end

