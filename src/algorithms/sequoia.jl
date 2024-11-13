export sequoia_solve!

"""
    sequoia_solve!(problem::SEQUOIA_pb, inner_solver, options, time, x, tk, iteration, inner_iterations)

Solve an optimization problem using SEQUOIA's modified bisection algorithm.

# Arguments
- `problem::SEQUOIA_pb`: The optimization problem to solve, represented as a `SEQUOIA_pb` instance.
- `inner_solver`: An `Optim` solver instance for solving subproblems (e.g., `Optim.LBFGS`).
- `options`: Optimization options for the inner solver (e.g., tolerances, maximum iterations).
- `time::Float64`: Accumulated computation time (updated in place).
- `x::Vector{Float64}`: The decision variable vector (updated in place).
- `tk::Float64`: Current threshold value for the objective (updated in place).
- `iteration::Int`: Current outer iteration count (updated in place).
- `inner_iterations::Int`: Accumulated inner solver iterations (updated in place).

# Returns
- Updated `time`, `x`, `tk`, `iteration`, and `inner_iterations`.

# Algorithm Overview
1. Initialize the penalty parameter `dk` and thresholds `tl` and `tu`.
2. Solve two subproblems:
   - Upper threshold problem: `r(x, tu, pb)`
   - Lower threshold problem: `r(x, tl, pb)`
3. Update the decision variable `x` based on convergence criteria.
4. Adjust the penalty parameter `dk` based on the results of the subproblems.
5. Repeat until convergence, maximum iterations, or time limit is reached.

# Convergence Criteria
- Converges if:
  1. The step size `dk` is below the tolerance, and
  2. Residual `rk` is within the residual tolerance.
- Terminates early if:
  - Step size `dk` becomes too small, or
  - The objective threshold `tk` falls below the minimum allowed value.

# Notes
- The function supports both SEQUOIA and CUTEst optimization problems.
- Solution history is stored in `problem.solution_history` if trace storage is enabled.
"""
function sequoia_solve!(problem::SEQUOIA_pb, inner_solver, options, time, x, tk, iteration, inner_iterations)

    dk=problem.solver_settings.solver_params[1];
    gamma=problem.solver_settings.solver_params[2];
    beta=problem.solver_settings.solver_params[3];
    tk=problem.objective(x)#min(tk,problem.objective(x))

    tu=tk;
    tl=tu-dk;

    if problem.cutest_nlp === nothing
        pb=problem
    else
        pb=problem.cutest_nlp
    end
    rk=r(x,tk,pb)

    while iteration < problem.solver_settings.max_iter_outer && time < problem.solver_settings.max_time_outer
        
        obj_aug_fnu = x -> r(x, tu, pb)
        grad_aug_fnu! = (g, x) -> r_gradient!(g, x, tu, pb)
        obj_aug_fnl = x -> r(x, tl, pb)
        grad_aug_fnl! = (g, x) -> r_gradient!(g, x, tl, pb)

        resultu = Optim.optimize(obj_aug_fnu, grad_aug_fnu!, x, inner_solver, options)
        resultl = Optim.optimize(obj_aug_fnl, grad_aug_fnl!, x, inner_solver, options)

        xl = resultl.minimizer;
        rl = resultl.minimum;
        xu = resultu.minimizer;
        ru = resultu.minimum;

        time += resultu.time_run + resultl.time_run
        inner_iterations += resultl.iterations + resultu.iterations

        if rl <= problem.solver_settings.resid_tolerance || rl<min(rk,tk)
            x=xl;
            rk=rl;
            tk=tl;
            dk*=gamma;
        elseif rk > problem.solver_settings.resid_tolerance && ru < rk
            x=xu;
            rk=ru;
            tk=tu;
            dk*=gamma;
        else
            dk*=beta;
        end

        if problem.solver_settings.store_trace
            step = SEQUOIA_Solution_step(iteration, dk, :unknown, resultl.time_run+resultu.time_run, resultl.iterations+resultu.iterations, x, rk, problem.gradient(x), problem.constraints(x), [tk], vcat(Optim.x_trace(resultl),Optim.x_trace(resultu)))
            add_iterate!(problem.solution_history, step)  # Add step to history
        end

        if dk<problem.solver_settings.cost_tolerance && rk <= problem.solver_settings.resid_tolerance         # Convergence check
            step = SEQUOIA_Solution_step(iteration, dk, :first_order, time, inner_iterations, x, rk, problem.gradient(x), problem.constraints(x), [tk])
            add_iterate!(problem.solution_history, step)  # Add step to history#

            return time, x, tu, iteration, inner_iterations
        elseif dk<10^(-16) && rk > problem.solver_settings.resid_tolerance #problem.solver_settings.cost_tolerance
            step = SEQUOIA_Solution_step(iteration, dk, :small_step, time, inner_iterations, x, rk, problem.gradient(x), problem.constraints(x), [tk])
            add_iterate!(problem.solution_history, step)  # Add step to history#

            return time, x, tu, iteration, inner_iterations
        elseif tk<problem.solver_settings.cost_min
            step = SEQUOIA_Solution_step(iteration, dk, :unbounded, time, inner_iterations, x, rk, problem.gradient(x), problem.constraints(x), [tk])
            add_iterate!(problem.solution_history, step)  # Add step to history#

            return time, x, tu, iteration, inner_iterations
        end

        if rk <= problem.solver_settings.resid_tolerance
            tu=tk;
        else 
            tu=tk+dk;
        end
        tl=tk-dk;

        # Increment iteration counter
        iteration += 1
    end

    if iteration >= problem.solver_settings.max_iter_outer
        solver_status = :max_iter
    elseif time >= problem.solver_settings.max_time_outer
        solver_status = :max_time
    else
        solver_status = :unknown
    end
    step = SEQUOIA_Solution_step(iteration, dk, solver_status, time, inner_iterations, x, rk, problem.gradient(x), problem.constraints(x), [tk])
    add_iterate!(problem.solution_history, step)  # Add step to history

    return time, x, tu, iteration, inner_iterations
end
