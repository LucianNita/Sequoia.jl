export qpm_solve!

# Quadratic Penalty Method (QPM) Implementation using Optim.jl
function qpm_solve!(problem::SEQUOIA_pb, inner_solver, options, time, x, previous_fval, iteration, inner_iterations)

    penalty_init = problem.solver_settings.solver_params[1]
    penalty_mult = problem.solver_settings.solver_params[2]
    damping_factor = problem.solver_settings.solver_params[3]
    penalty_param = penalty_init  # Set initial penalty parameter

    result = nothing;
    constraint_violation=nothing;

    while iteration < problem.solver_settings.max_iter_outer && time < problem.solver_settings.max_time_outer
        
        if problem.cutest_nlp === nothing
            # Use Optim.jl to minimize the augmented objective function
            obj_aug_fn = x -> qpm_obj(x, penalty_param, problem)
            grad_aug_fn! = (g, x) -> qpm_grad!(g, x, penalty_param, problem)
        else 
            obj_aug_fn = x -> qpm_obj(x, penalty_param, problem.cutest_nlp)
            grad_aug_fn! = (g, x) -> qpm_grad!(g, x, penalty_param, problem.cutest_nlp)
        end

        # Solve the unconstrained subproblem using the inner solver
        result = Optim.optimize(obj_aug_fn, grad_aug_fn!, x, inner_solver, options)

        # Extract the optimized solution from the subproblem
        x = result.minimizer

        # Compute constraint violation and objective value
        if problem.cutest_nlp === nothing
            constraint_violation = exact_constraint_violation(x, problem)
        else
            constraint_violation = exact_constraint_violation(x, problem.cutest_nlp)
        end

        fval = result.minimum  # Objective function value
        time += result.time_run

        conv = constraint_violation < problem.solver_settings.resid_tolerance && Optim.converged(result)
        # Check for convergence using improved criteria
        if !isnothing(problem.solver_settings.cost_tolerance)
            conv = conv && abs(fval - previous_fval) < problem.solver_settings.cost_tolerance
        end
        if conv
            if problem.solver_settings.store_trace
                x_tr=Optim.x_trace(result);
            else
                x_tr=nothing;
            end
            step = SEQUOIA_Solution_step(iteration, abs(fval - previous_fval), :first_order, result.time_run, result.iterations, x, fval, problem.gradient(x), problem.constraints(x), [penalty_param], x_tr)
            add_iterate!(problem.solution_history, step)  # Add step to history

            previous_fval=fval;

            return time, x, previous_fval, iteration
        end

        if problem.solver_settings.store_trace
            step = SEQUOIA_Solution_step(iteration, abs(fval - previous_fval), :unknown, result.time_run, result.iterations, x, fval, problem.gradient(x), problem.constraints(x), [penalty_param],Optim.x_trace(result) )
            add_iterate!(problem.solution_history, step)  # Add step to history
        end

        # Update previous function value for the next iteration
        previous_fval = fval

        # Penalty parameter adjustment based on constraint violation
        if problem.solver_settings.solver_params[4] == 0
            penalty_param *= penalty_mult
        else
            penalty_param *= min(max(1, constraint_violation / problem.solver_settings.resid_tolerance), damping_factor)
        end 
        
        # Increment the iteration counter
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
        step = SEQUOIA_Solution_step(iteration, abs(result.minimum - previous_fval), solver_status, result.time_run, result.iterations, result.minimizer, result.minimum, problem.gradient(x), problem.constraints(result.minimizer), [penalty_param])
        add_iterate!(problem.solution_history, step)  # Add step to history    
    end
    
    return time, x, previous_fval, iteration, inner_iterations
end
