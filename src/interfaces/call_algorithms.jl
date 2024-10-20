using ForwardDiff

function solve(problem::SEQUOIA_pb)
    # Extract key components from SEQUOIA_pb problem

    # Objective function
    obj_fn = problem.objective

    # Gradient: Use provided gradient or compute it automatically
    grad_fn = problem.gradient
    if grad_fn === nothing
        grad_fn = x -> ForwardDiff.gradient(obj_fn, x)
        println("Using automatic differentiation to compute the gradient of the objective.")
    end

    # Constraint function and Jacobian: Use provided ones or compute Jacobian automatically
    cons_fn = problem.constraints
    if cons_fn === nothing
        cons_fn = x -> []
    end

    jac_fn = problem.jacobian
    if jac_fn === nothing && cons_fn !== nothing
        jac_fn = x -> ForwardDiff.jacobian(cons_fn, x)
        println("Using automatic differentiation to compute the Jacobian of the constraints.")
    end

    # Lower and upper bounds for the constraints 
    lb = problem.l_bounds
    ub = problem.u_bounds

    # Equality and inequality constraints
    eq_indices = problem.eqcon
    ineq_indices = problem.ineqcon

    # Initial guess
    x0 = problem.x0
    if isempty(x0)
        warning("Initial guess `x0` must be provided. Setting a default guess to zero.")
        x0 = zeros(problem.nvar)
    end

    # Extract solver settings from SEQUOIA_Settings
    settings = problem.solver_settings

    if settings.outer_method isa QPM
        tol = settings.resid_tolerance  # Residual tolerance for convergence
        max_iter = settings.max_iter  # Maximum number of iterations
        penalty_mult = settings.step_size === nothing ? 10.0 : settings.step_size  # Use step_size if set; otherwise, default
        damping_factor = penalty_mult

        # Inner solver from SEQUOIA_Settings
        inner_solver = choose_inner_solver(settings.inner_solver)

        # Call the QPM algorithm with extracted data
        x_opt, final_penalty, step_size, num_inner_iterations, solver_status = qpm_solve(
            obj_fn, grad_fn, cons_fn, jac_fn, lb, ub, eq_indices, ineq_indices, x0, inner_solver;
            penalty_mult=penalty_mult, 
            tol=tol, 
            max_iter=max_iter, 
            damping_factor=damping_factor
        )

        # Update SEQUOIA_Solution_step
        grad_val = grad_fn(x_opt)
        c_val = cons_fn(x_opt)
        convergence_metric = norm(grad_val)  # Example of a convergence metric (norm of gradient)
        outer_iter = max_iter  # Just an example; adjust based on actual outer iteration count

        # Update solution and solution history
        solution_step = SEQUOIA_Solution_step(
            x = x_opt,
            fval = obj_fn(x_opt),
            gval = grad_val,
            cval = c_val,
            step_size = step_size,
            convergence_metric = convergence_metric,
            outer_iteration_number = outer_iter,
            num_inner_iterations = num_inner_iterations,
            inner_comp_time = 0.0,  # You can measure time taken if necessary
            solver_status = solver_status
        )

        # Update the problem's current solution and solution history
        problem.solution = solution_step
        add_step!(problem.solution_history, solution_step)
    end
end

# Helper function to map SEQUOIA_pb inner solvers to Optim.jl solvers
function choose_inner_solver(inner_solver::InnerSolver)
    if inner_solver isa LBFGS
        return Optim.LBFGS()
    elseif inner_solver isa BFGS
        return Optim.BFGS()
    elseif inner_solver isa Newton
        return Optim.Newton()
    elseif inner_solver isa GradientDescent
        return Optim.GradientDescent()
    elseif inner_solver isa NelderMead
        return Optim.NelderMead()
    else
        error("Unknown inner solver: $inner_solver. Make sure you use one of the accepted and tested solvers.")
    end
end
