export feasibility_solve!

"""
    feasibility_solve!(problem::SEQUOIA_pb, inner_solver, options, time, x, previous_fval, inner_iterations)

Solve the feasibility subproblem for a `SEQUOIA_pb` instance using an inner solver.

# Arguments
- `problem::SEQUOIA_pb`: The SEQUOIA optimization problem instance.
- `inner_solver`: The solver to use for the unconstrained subproblem (e.g., `Optim.LBFGS`).
- `options`: A set of options for the solver (e.g., tolerances, maximum iterations).
- `time::Float64`: The accumulated runtime, which will be updated by the function.
- `x::Vector{Float64}`: The decision variable vector, modified in place to store the current solution.
- `previous_fval::Float64`: The previous objective function value, updated by the function.
- `inner_iterations::Int`: The accumulated number of inner solver iterations, updated by the function.

# Returns
- Updated `time`, `x`, `previous_fval`, and `inner_iterations`.

# Notes
- If `problem.cutest_nlp` is not `nothing`, the function assumes the problem uses the CUTEst NLP interface.
- The function solves an unconstrained optimization problem defined by the residual function `r0` and its gradient `r0_gradient!`.
- The solution history is updated with a `SEQUOIA_Solution_step` object after solving the subproblem.
- The solver status is determined based on the residual norm (`result.minimum`) and convergence status.
"""
function feasibility_solve!(problem::SEQUOIA_pb, inner_solver, options, time, x, previous_fval, inner_iterations)
    if problem.cutest_nlp === nothing
        obj_aug_fn = x -> r0(x,problem)
        grad_aug_fn! = (g, x) -> r0_gradient!(g, x, problem)
    else
        obj_aug_fn = x -> r0(x,problem.cutest_nlp)
        grad_aug_fn! = (g, x) -> r0_gradient!(g, x, problem.cutest_nlp)
    end

    # Solve the unconstrained subproblem
    result = Optim.optimize(obj_aug_fn, grad_aug_fn!, x, inner_solver, options)

    x .= result.minimizer
    time+=result.time_run;
    previous_fval = problem.objective(x);
    inner_iterations += result.iterations
    solver_status = ( result.minimum <= problem.solver_settings.resid_tolerance && Optim.converged(result) ) ? :small_residual : :infeasible  # Solver status

    # Create a SEQUOIA_Solution_step and save it to history
    step = SEQUOIA_Solution_step(0, result.minimum, solver_status, result.time_run, result.iterations , x, result.minimum, problem.gradient(x), problem.constraints(x))
    add_iterate!(problem.solution_history, step)  # Add step to history

    return time, x, previous_fval, inner_iterations
end
