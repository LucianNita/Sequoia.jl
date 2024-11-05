export feasibility_solve!

function feasibility_solve!(problem::SEQUOIA_pb, inner_solver, options, time, x, previous_fval)
    if problem.cutest_nlp === nothing
        obj_aug_fn = x -> r0(x,problem)
        grad_aug_fn! = (g, x) -> r0_gradient!(g, x, problem)
    else
        obj_aug_fn = x -> r0(x,problem.cutest_nlp)
        grad_aug_fn! = (g, x) -> r0_gradient!(g, x, problem.cutest_nlp)
    end

    # Solve the unconstrained subproblem
    result = Optim.optimize(obj_aug_fn, grad_aug_fn!, x, inner_solver, options)

    x = result.minimizer
    time+=result.time_run;
    previous_fval=result.minimum;

    solver_status = Optim.converged(result) ? :small_residual : :infeasible  # Solver status

    # Create a SEQUOIA_Solution_step and save it to history
    step = SEQUOIA_Solution_step(0, result.minimum, solver_status, result.time_run, result.iterations , x, result.minimum, problem.gradient(x), problem.constraints(x))
    add_iterate!(problem.solution_history, step)  # Add step to history

    return time, x, previous_fval
end
