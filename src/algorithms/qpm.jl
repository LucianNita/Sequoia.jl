export qpm_solve!

"""
    qpm_solve!(problem::SEQUOIA_pb, inner_solver, options, time, x, previous_fval, iteration, inner_iterations)

Solve an optimization problem using the Quadratic Penalty Method (QPM).

# Arguments
- `problem`: A `SEQUOIA_pb` optimization problem instance.
- `inner_solver`: The unconstrained solver from `Optim.jl` to solve subproblems (e.g., `Optim.LBFGS`).
- `options`: Solver options for the inner solver.
- `time`: Total elapsed time for the solver (updated in-place).
- `x`: The decision variable vector (updated in-place).
- `previous_fval`: The previous objective function value (updated in-place).
- `iteration`: The current outer iteration count (updated in-place).
- `inner_iterations`: The total number of inner iterations (updated in-place).

# Returns
- Updated `time`, `x`, `previous_fval`, `iteration`, and `inner_iterations`.

# Notes
- The method iteratively solves unconstrained subproblems using the penalty parameter.
- Convergence is checked based on residual tolerance and optional cost improvement.
- Supports both `SEQUOIA_pb` and `CUTEstModel` problems transparently.
- Tracks intermediate solutions using `SEQUOIA_Solution_step` objects stored in `problem.solution_history`.
"""
function qpm_solve!(problem::SEQUOIA_pb, inner_solver, options, time, x, previous_fval, iteration, inner_iterations) # Quadratic Penalty Method (QPM) Implementation using Optim.jl

    penalty_init = problem.solver_settings.solver_params[1]
    penalty_mult = problem.solver_settings.solver_params[2]
    damping_factor = problem.solver_settings.solver_params[3]
    penalty_param = penalty_init  # Set initial penalty parameter

    fval=previous_fval;
    if problem.cutest_nlp === nothing
        pb=problem;
    else
        pb=problem.cutest_nlp
    end

    while iteration < problem.solver_settings.max_iter_outer && time < problem.solver_settings.max_time_outer
        
        obj_aug_fn = x -> qpm_obj(x, penalty_param, pb)
        grad_aug_fn! = (g, x) -> qpm_grad!(g, x, penalty_param, pb)

        # Solve the unconstrained subproblem using the inner solver
        result = Optim.optimize(obj_aug_fn, grad_aug_fn!, x, inner_solver, options)

        # Extract the optimized solution from the subproblem
        x .= result.minimizer

        fval = result.minimum  # Objective function value
        time += result.time_run
        inner_iterations += result.iterations

        conv = r0(x, pb) <= problem.solver_settings.resid_tolerance && Optim.converged(result)
        if !isnothing(problem.solver_settings.cost_tolerance) # Check for convergence using improved criteria
            conv = conv && abs(fval - previous_fval) < problem.solver_settings.cost_tolerance
        end
        if conv
            problem.solver_settings.store_trace ? x_tr=Optim.x_trace(result) : x_tr=nothing
            step = SEQUOIA_Solution_step(iteration, abs(fval - previous_fval), :first_order, result.time_run, result.iterations, x, fval, problem.gradient(x), problem.constraints(x), [penalty_param], x_tr)
            add_iterate!(problem.solution_history, step)  # Add step to history

            previous_fval=fval;
            return time, x, previous_fval, iteration, inner_iterations
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
            penalty_param *= min(max(1, r0(x, pb) / problem.solver_settings.resid_tolerance), damping_factor)
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
        step = SEQUOIA_Solution_step(iteration, abs(fval - previous_fval), solver_status, time, inner_iterations, x, fval, problem.gradient(x), problem.constraints(x), [penalty_param])
        add_iterate!(problem.solution_history, step)  # Add step to history    
    end
    
    return time, x, previous_fval, iteration, inner_iterations
end
