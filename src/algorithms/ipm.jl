export ipm_solve!

"""
    ipm_solve!(problem::SEQUOIA_pb, inner_solver, options, time, x, previous_fval, iteration, inner_iterations)

Solve a nonlinear optimization problem using the Interior Point Method (IPM).

# Arguments
- `problem::SEQUOIA_pb`: The SEQUOIA problem to solve.
- `inner_solver`: The unconstrained optimization solver (e.g., `Optim.LBFGS`).
- `options`: Options for the inner solver (e.g., maximum iterations, tolerance).
- `time`: Accumulated runtime (in seconds) of the solver.
- `x`: Initial guess for the decision variables.
- `previous_fval`: Initial objective function value.
- `iteration`: Current iteration number of the outer loop.
- `inner_iterations`: Accumulated number of iterations in the inner solver.

# Returns
- `time`: Total runtime of the solver.
- `x`: Final solution for the decision variables.
- `previous_fval`: Final objective function value.
- `iteration`: Final iteration count of the outer loop.
- `inner_iterations`: Total iterations in the inner solver.

# Notes
- The function solves the problem by iteratively minimizing the IPM objective (`ipm_obj`) using the given inner solver.
- Barrier methods are employed to ensure feasibility for inequality constraints.
- The penalty parameter is adjusted dynamically to balance constraint violation and convergence.
- Intermediate results are stored in `problem.solution_history` for traceability.
"""
function ipm_solve!(problem::SEQUOIA_pb, inner_solver, options, time, x, previous_fval, iteration, inner_iterations)

    # Initialize variables
    penalty_init=problem.solver_settings.solver_params[1];
    penalty_mult=problem.solver_settings.solver_params[2];
    damping_factor=problem.solver_settings.solver_params[3];
    penalty_param = penalty_init  # Set initial penalty parameter
    λ = problem.solver_settings.solver_params[5:end];  # Initialize or warm-start Lagrange multipliers
    n_con=length(λ);
    if problem.cutest_nlp === nothing
        pb = problem;
    else
        pb = problem.cutest_nlp;
        con = res(x,pb)
    end
    

    while iteration < problem.solver_settings.max_iter_outer && time < problem.solver_settings.max_time_outer

        obj_aug_fn = x -> ipm_obj(x, penalty_param, pb)
        grad_aug_fn! = (g, x) -> ipm_grad!(g, x, penalty_param, pb)

        # Solve the unconstrained subproblem
        result = Optim.optimize(obj_aug_fn, grad_aug_fn!, vcat(x,λ), inner_solver, options)

        # Extract the optimized solution from the subproblem
        x = result.minimizer[1:problem.nvar]
        λ = result.minimizer[problem.nvar+1:problem.nvar+n_con]
        fval = result.minimum
        time+= result.time_run  # Computation time
        inner_iterations += result.iterations
        
        conv = Optim.converged(result) && r0(x,pb) ≤ problem.solver_settings.resid_tolerance && abs(fval - previous_fval) < problem.solver_settings.cost_tolerance 
        if conv
            problem.solver_settings.store_trace ? x_tr=Optim.x_trace(result) : x_tr=nothing
            step = SEQUOIA_Solution_step(iteration, abs(fval - previous_fval), :first_order, result.time_run, result.iterations, x, fval, problem.gradient(x), problem.constraints(x),  vcat(penalty_param,λ), x_tr)
            add_iterate!(problem.solution_history, step)  # Add step to history

            previous_fval=fval;
            return time, x, previous_fval, iteration, inner_iterations
        end

        if problem.solver_settings.store_trace
            step = SEQUOIA_Solution_step(iteration, abs(fval - previous_fval), :unknown, result.time_run, result.iterations, x, fval, problem.gradient(x), problem.constraints(x), vcat(penalty_param,λ),Optim.x_trace(result) )
            add_iterate!(problem.solution_history, step)  # Add step to history
        end

        # Update previous function value for the next iteration
        previous_fval = fval
      
        # Penalty parameter adjustment based on constraint violation
        if problem.solver_settings.solver_params[4]==0
            penalty_param *= penalty_mult;
        else
            penalty_param *= max(min(1, problem.solver_settings.resid_tolerance / r0(x,pb)), damping_factor);
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
        step = SEQUOIA_Solution_step(iteration, abs(fval - previous_fval), solver_status, time, inner_iterations, result.minimizer[1:problem.nvar], fval, problem.gradient(x[1:problem.nvar]), problem.constraints(result.minimizer[1:problem.nvar]), vcat(penalty_param,λ))
        add_iterate!(problem.solution_history, step)  # Add step to history
    end    

    return time, x, previous_fval, iteration, inner_iterations
end

