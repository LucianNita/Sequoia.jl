# Quadratic Penalty Method (QPM) Implementation using Optim.jl
function qpm_solve(problem, inner_solver)

    penalty_init=problem.solver_settings.solver_params[1];
    penalty_mult=problem.solver_settings.solver_params[2];
    damping_factor=problem.solver_settings.solver_params[3];
    rtol=problem.solver_settings.resid_tolerance;
    x=problem.x0;

    eq_indices=problem.eqcon;
    ineq_indices=problem.ineqcon;

    iteration = 1  # Initialize iteration counter
    time = 0 # Initialization of computational time
    penalty_param = penalty_init  # Set initial penalty parameter
    solution_history = SEQUOIA_History();  # Initialize the solution history

    # Helper function: Augmented objective with penalty for equality and inequality constraints
    function augmented_objective(x, penalty_param)
        constraint_val = problem.constraints(x)
        eq_penalty_term = 0.0
        ineq_penalty_term = 0.0

        # Handle equality constraints (penalty applied if violated)
        for i in eq_indices
            eq_penalty_term += 0.5 * penalty_param * (constraint_val[i])^2
        end

        # Handle inequality constraints (penalty applied if outside bounds)
        for i in ineq_indices
            if constraint_val[i] > 0.0
                ineq_penalty_term += 0.5 * penalty_param * (constraint_val[i])^2 #can be replaced with max(0,fn)
            end
        end

        # Return the augmented objective (original objective + penalties)
        return problem.objective(x) + eq_penalty_term + ineq_penalty_term
    end

    # Helper function: In-place gradient of the augmented objective with penalty
    function augmented_gradient!(grad_storage, x, penalty_param)
        constraint_val = problem.constraints(x)
        grad_obj = problem.gradient(x)
        
        # Initialize penalty gradients
        grad_eq_penalty = zeros(length(x))
        grad_ineq_penalty = zeros(length(x))

        # Handle equality constraints
        for i in eq_indices
            grad_eq_penalty += penalty_param * problem.jacobian(x)[i, :] * (constraint_val[i])
        end

        # Handle inequality constraints
        for i in ineq_indices
            if constraint_val[i] > 0.0
                grad_ineq_penalty += penalty_param * problem.jacobian(x)[i, :] * (constraint_val[i])
            end
        end

        # Update the gradient storage with the objective gradient + penalties
        grad_storage .= grad_obj .+ grad_eq_penalty .+ grad_ineq_penalty
    end

    # Function to compute the total constraint violation (used for adaptive penalty updates)
    function compute_constraint_violation(x)
        constraint_val = problem.constraints(x)
        eq_violation = norm(constraint_val[eq_indices])  # Equality violation
        ineq_violation = norm(max.(0.0, constraint_val[ineq_indices]))  # Inequality violation
        return eq_violation + ineq_violation
    end

    while iteration < problem.solver_settings.max_iter_outer && time < problem.solver_settings.max_time_outer
        
        # Use Optim.jl to minimize the augmented objective function
        obj_aug_fn = x -> augmented_objective(x, penalty_param)
        grad_aug_fn! = (g, x) -> augmented_gradient!(g, x, penalty_param)
        
        # Set Optim options
        if problem.solver_settings.conv_crit==:GradientNorm
            options = Optim.Options(g_tol=rtol, store_trace=true, extended_trace=true, show_trace=false)
        elseif problem.solver_settings.conv_crit==:MaxIterations
            options = Optim.Options(g_tol=rtol, iterations=problem.solver_settings.max_iter_inner, store_trace=true, extended_trace=true, show_trace=false)
        elseif problem.solver_settings.conv_crit==:MaxTime
            options = Optim.Options(g_tol=rtol, time_limit=problem.solver_settings.max_time_inner, store_trace=true, extended_trace=true, show_trace=false)
        elseif problem.solver_settings.conv_crit==:CombinedCrit
            options = Optim.Options(g_tol=rtol, iterations=problem.solver_settings.max_iter_inner, time_limit=problem.solver_settings.max_time_inner, store_trace=true, extended_trace=true, show_trace=false)
        else
            throw(ArgumentError("Invalid convergence criterion: $conv_crit. Valid criteria are: $(join(valid_convergence_criterias, ", "))."))
        end
        # Solve the unconstrained subproblem using the inner solver
        result = Optim.optimize(obj_aug_fn, grad_aug_fn!, x, inner_solver, options)

        # Extract the optimized solution from the subproblem
        x = result.minimizer
        
        # Compute the constraint violation for adaptive updates
        constraint_violation = compute_constraint_violation(x)

        # Save a SEQUOIA_Solution_step after each optimize call
        fval = result.minimum  # Objective function value
        gval = problem.gradient(x)  # Gradient of the objective
        cval = problem.constraints(x)  # Constraint values
        solver_status = Optim.converged(result) ? :success : :not_converged  # Solver status
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



