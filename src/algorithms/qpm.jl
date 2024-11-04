# Quadratic Penalty Method (QPM) Implementation using Optim.jl
function qpm_solve(problem::SEQUOIA_pb, inner_solver,options)

    penalty_init=problem.solver_settings.solver_params[1];
    penalty_mult=problem.solver_settings.solver_params[2];
    damping_factor=problem.solver_settings.solver_params[3];
    rtol=problem.solver_settings.resid_tolerance;
    x=problem.x0;

    iteration = 1  # Initialize iteration counter
    time = 0.0 # Initialization of computational time
    penalty_param = penalty_init  # Set initial penalty parameter
    solution_history = SEQUOIA_History();  # Initialize the solution history

    while iteration < problem.solver_settings.max_iter_outer && time < problem.solver_settings.max_time_outer
        
        if problem.cutest_nlp === nothing
            # Use Optim.jl to minimize the augmented objective function
            obj_aug_fn = x -> qpm_obj(x, penalty_param, problem)
            grad_aug_fn! = (g, x) -> qpm_grad!(g, x, penalty_param, problem)
        else 
            obj_aug_fn = x -> qpm_obj(x, penalty_param, problem.cutest_nlp)
            grad_aug_fn! = (g, x) -> qpm_grad!(g, x, penalty_param,problem.cutest_nlp)
        end

        

        # Solve the unconstrained subproblem using the inner solver
        result = Optim.optimize(obj_aug_fn, grad_aug_fn!, x, inner_solver, options)

        # Extract the optimized solution from the subproblem
        x = result.minimizer
        
        if problem.cutest_nlp === nothing
            # Compute the constraint violation for adaptive updates
            constraint_violation = exact_constraint_violation(x,problem)
        else
            constraint_violation = exact_constraint_violation(x,problem.cutest_nlp)
        end

        # Save a SEQUOIA_Solution_step after each optimize call
        fval = result.minimum  # Objective function value
        gval = problem.gradient(x)  # Gradient of the objective
        cval = problem.constraints(x)  # Constraint values
        solver_status = Optim.converged(result) ? :first_order : :unkown  # Solver status
        inner_comp_time = result.time_run  # Computation time
        num_inner_iterations = result.iterations  # Number of inner iterations
        x_tr = Optim.x_trace(result)  # This returns the history of iterates

        conv=constraint_violation;
        # Create a SEQUOIA_Solution_step and save it to history
        step = SEQUOIA_Solution_step(iteration, conv, solver_status, inner_comp_time, num_inner_iterations, x, fval, gval, cval, [penalty_param], x_tr)
        add_iterate!(solution_history, step)  # Add step to history

        # Check for convergence based on the constraint violation   #Can be based on f difference too 
        if conv < rtol   
            println("Converged after $(iteration) iterations.")
            return solution_history
        end

        if problem.solver_settings.solver_params[4]==0
            penalty_param *= penalty_mult;
        else
            penalty_param *= min(max(1, constraint_violation / rtol), damping_factor);
        end 
        
        # Increment the iteration counter & time
        iteration += 1
        time+=inner_comp_time;
    end

    
    return solution_history
end



