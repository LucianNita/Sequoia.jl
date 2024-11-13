export alm_solve!

"""
    alm_solve!(problem::SEQUOIA_pb, inner_solver, options, time, x, previous_fval, iteration, inner_iterations)

Solve a constrained optimization problem using the Augmented Lagrangian Method (ALM).

# Arguments
- `problem::SEQUOIA_pb`: A `SEQUOIA_pb` problem instance containing the objective, constraints, and settings.
- `inner_solver`: The unconstrained solver (e.g., `Optim.LBFGS`) for solving subproblems.
- `options`: Optimization options for the `inner_solver` (e.g., tolerance, maximum iterations).
- `time`: Accumulated time spent in solving the problem (updated in-place).
- `x`: Initial guess for the decision variables (updated in-place).
- `previous_fval`: Previous objective value, used for convergence checks (updated in-place).
- `iteration`: Current iteration number of the outer ALM loop (updated in-place).
- `inner_iterations`: Total inner solver iterations (updated in-place).

# Returns
- `time`: Total time spent in solving the problem.
- `x`: Final optimized decision variables.
- `previous_fval`: Final objective value.
- `iteration`: Total number of outer iterations performed.
- `inner_iterations`: Total number of inner solver iterations performed.

# Notes
- The augmented Lagrangian objective is minimized iteratively using `inner_solver`.
- Lagrange multipliers (`λ`) are updated in each iteration using `update_lag_mult!`.
- Penalty parameters are adjusted dynamically based on the residuals and the damping factor.
- Convergence is determined by:
  1. Residual tolerance (`r0(x, pb)`).
  2. Cost improvement tolerance (if specified).
  3. Solver convergence status (`Optim.converged`).
"""
function alm_solve!(problem::SEQUOIA_pb, inner_solver, options, time, x, previous_fval, iteration, inner_iterations) 

    # Initialize variables
    penalty_init=problem.solver_settings.solver_params[1];
    penalty_mult=problem.solver_settings.solver_params[2];
    damping_factor=problem.solver_settings.solver_params[3];
    penalty_param = penalty_init  # Set initial penalty parameter
    λ = problem.solver_settings.solver_params[5:end];  # Initialize or warm-start Lagrange multipliers
    
    fval=previous_fval;
    if problem.cutest_nlp === nothing
        pb=problem;
    else
        pb=problem.cutest_nlp
    end

    while iteration < problem.solver_settings.max_iter_outer && time < problem.solver_settings.max_time_outer

    
        obj_aug_fn = x -> auglag_obj(x, penalty_param, λ, pb)
        grad_aug_fn! = (g, x) -> auglag_grad!(g, x, penalty_param, λ, pb)

        # Solve the unconstrained subproblem
        result = Optim.optimize(obj_aug_fn, grad_aug_fn!, x, inner_solver, options)

        # Extract the optimized solution from the subproblem
        x = result.minimizer

        # Save a SEQUOIA_Solution_step after each optimize call
        fval = result.minimum  # Objective function value
        time+= result.time_run  # Computation time
        inner_iterations += result.iterations

        if problem.solver_settings.store_trace
            step = SEQUOIA_Solution_step(iteration, abs(fval - previous_fval), :unknown, result.time_run, result.iterations, x, fval, problem.gradient(x), problem.constraints(x), vcat(penalty_param,λ),Optim.x_trace(result) )
            add_iterate!(problem.solution_history, step)  # Add step to history
        end
        
        conv = r0(x, pb) < problem.solver_settings.resid_tolerance && Optim.converged(result)
        # Check for convergence using improved criteria
        if !isnothing(problem.solver_settings.cost_tolerance)
            conv = conv && abs(fval - previous_fval) < problem.solver_settings.cost_tolerance
        end
        if conv
            step = SEQUOIA_Solution_step(iteration, abs(fval - previous_fval), :first_order, time, inner_iterations, x, fval, problem.gradient(x), problem.constraints(x),  vcat(penalty_param,λ))
            add_iterate!(problem.solution_history, step)  # Add step to history

            previous_fval=fval;
            return time, x, previous_fval, iteration, inner_iterations
        end
        
        # Update previous function value for the next iteration
        previous_fval = fval
        update_lag_mult!(x, penalty_param, λ, problem) # Update Lagrange multipliers
      

        if problem.solver_settings.solver_params[4]==0
            penalty_param *= penalty_mult;
        else
            penalty_param *= min(max(1, r0(x, pb) / problem.solver_settings.resid_tolerance), damping_factor);
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
    step = SEQUOIA_Solution_step(iteration, abs(fval - previous_fval), solver_status, time, inner_iterations, x, fval, problem.gradient(x), problem.constraints(x), vcat(penalty_param,λ))
    add_iterate!(problem.solution_history, step)  # Add step to history    

    return time, x, previous_fval, iteration, inner_iterations
end


