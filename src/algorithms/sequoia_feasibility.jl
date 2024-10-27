using SparseArrays

export sequoia_feasibility_cutest

function sequoia_feasibility_cutest(problem::CUTEstModel, inner_solver)
    # Extract initial parameters
    x = problem.meta.x0  # Initial guess
    rtol=10^-6;
    max_iter_inner = 1000

    iteration = 0
    solution_history = SEQUOIA_History();

    obj_aug_fn = x -> r0(x,problem)
    grad_aug_fn! = (g, x) -> r0_gradient!(g, x, problem)
    
    # Optim.jl options for inner solve
    options = Optim.Options(g_tol=rtol, iterations=max_iter_inner, store_trace=true, extended_trace=true, show_trace=false)
    
    # Solve the unconstrained subproblem
    result = Optim.optimize(obj_aug_fn, grad_aug_fn!, x, inner_solver, options)

    # Extract the optimized solution from the subproblem
    x = result.minimizer
        
    # Compute the constraint violation for adaptive updates
    constraint_violation = exact_constraint_violation(x,problem)

    # Save a SEQUOIA_Solution_step after each optimize call
    fval = result.minimum  # Objective function value
    gval = grad(problem,x)  # Gradient of the objective
    cval = cons(problem,x)  # Constraint values
    solver_status = Optim.converged(result) ? :success : :not_converged  # Solver status
    inner_comp_time = result.time_run  # Computation time
    num_inner_iterations = result.iterations  # Number of inner iterations
    x_tr = Optim.x_trace(result)  # This returns the history of iterates

    conv=constraint_violation;

    # Create a SEQUOIA_Solution_step and save it to history
    step = SEQUOIA_Solution_step(iteration, conv, solver_status, inner_comp_time, num_inner_iterations, x, fval, gval, cval, [obj(problem,x), fval], x_tr)
    add_iterate!(solution_history, step)  # Add step to history

    return solution_history
end
