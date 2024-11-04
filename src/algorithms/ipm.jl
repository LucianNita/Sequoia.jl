# Interior Point Method (IPM) implementation
function ipm_solve(problem::SEQUOIA_pb, inner_solver, options)

    # Initialize variables
    penalty_init=problem.solver_settings.solver_params[1];
    penalty_mult=problem.solver_settings.solver_params[2];
    damping_factor=problem.solver_settings.solver_params[3];
    rtol=problem.solver_settings.resid_tolerance;
    x=vcat(problem.x0,ones(length(problem.ineqcon)));

    iteration = 1  # Initialize iteration counter
    time = 0.0 # Initialization of computational time
    penalty_param = penalty_init  # Set initial penalty parameter
    λ = problem.solver_settings.solver_params[5:end];  # Initialize or warm-start Lagrange multipliers
    solution_history = SEQUOIA_History();  # Initialize the solution history

    

    while iteration < problem.solver_settings.max_iter_outer && time < problem.solver_settings.max_time_outer
        if problem.cutest_nlp === nothing
            obj_aug_fn = x -> ipm_obj(x, penalty_param, λ, problem)
            grad_aug_fn! = (g, x) -> ipm_grad!(g, x, penalty_param, λ, problem)
        else
            obj_aug_fn = x -> ipm_obj(x, penalty_param, λ, problem.cutest_nlp)
            grad_aug_fn! = (g, x) -> ipm_grad!(g, x, penalty_param, λ, problem.cutest_nlp)
        end

        # Solve the unconstrained subproblem
        result = Optim.optimize(obj_aug_fn, grad_aug_fn!, x, inner_solver, options)
        # Extract the optimized solution from the subproblem
        x = result.minimizer

        if problem.cutest_nlp === nothing
            # Compute the constraint violation for adaptive updates
            constraint_violation = exact_constraint_violation(x[1:problem.nvar],problem)
        else
            constraint_violation = exact_constraint_violation(x[1:problem.nvar],problem.cutest_nlp)
        end

        # Save a SEQUOIA_Solution_step after each optimize call
        fval = result.minimum  # Objective function value
        gval=zeros(length(x))
        ipm_grad!(gval, x, penalty_param, λ, problem)
        #gval = problem.gradient(x)  # Gradient of the objective
        cval = problem.constraints(x[1:problem.nvar])  # Constraint values
        solver_status = Optim.converged(result) ? :success : :not_converged  # Solver status
        inner_comp_time = result.time_run  # Computation time
        num_inner_iterations = result.iterations  # Number of inner iterations
        x_tr = Optim.x_trace(result)  # This returns the history of iterates #result.g_residual 

        conv=constraint_violation;

        # Create a SEQUOIA_Solution_step and save it to history
        step = SEQUOIA_Solution_step(iteration, conv, solver_status, inner_comp_time, num_inner_iterations, x, fval, gval, cval, vcat(penalty_param,λ), x_tr)
        add_iterate!(solution_history, step)  # Add step to history
        # Check for convergence
        if constraint_violation < rtol
            println("Converged after $iteration iterations.")
            return solution_history
        end

        # Update Lagrange multipliers
        update_ipm_mult!(x, penalty_param, λ, problem)


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


##########################################################################################################################################
#=
# KKT System Solver with equality constraints and quadratic slack variables
function solve_kkt_quadratic_slack(x, s, λ, ν, μ)
    # Assemble the KKT system
    n = length(x)
    m = length(s)
    p = length(h(x))  # Number of equality constraints

    # Gradient of Lagrangian (Stationarity)
    Lx = ∇f(x) + ∇g(x)' * λ + ∇h(x)' * ν

    # Primal feasibility: g_i(x) + s_i^2 = 0 and h_j(x) = 0
    primal_feas_g = g(x) + s.^2
    primal_feas_h = h(x)

    # Build the system of equations (KKT system)
    KKT_matrix = [zeros(n, n) ∇g(x)' ∇h(x)' zeros(n, m);
                  ∇g(x) zeros(m, m) zeros(m, p) diagm(2 .* s);
                  ∇h(x) zeros(p, m) zeros(p, p) zeros(p, m);
                  zeros(m, n) diagm(2 .* s) zeros(m, p) I(m)]

    KKT_rhs = [-Lx;
               -primal_feas_g;
               -primal_feas_h;
               zeros(m)]

    # Solve the KKT system for search directions Δx, Δλ, Δν, Δs
    Δ = KKT_matrix \ KKT_rhs

    Δx = Δ[1:n]
    Δλ = Δ[n+1:n+m]
    Δν = Δ[n+m+1:n+m+p]
    Δs = Δ[n+m+p+1:end]

    return Δx, Δλ, Δν, Δs
end

# Line search function
function line_search(x, s, λ, ν, Δx, Δλ, Δν, Δs)
    t = 1.0

    # Backtracking line search to ensure objective decreases
    while f(x + t * Δx) > f(x) + α * t * dot(∇f(x), Δx)
        t *= β
    end

    return t
end

# Interior Point Method with Quadratic Slack and Equality Constraints
function interior_point_quadratic_slack_method(x0, s0, λ0, ν0, μ_initial)
    x = x0
    s = s0
    λ = λ0
    ν = ν0
    μ_param = μ_initial

    # Iterate until convergence
    while μ_param > tolerance
        # Solve the KKT system using Newton's method with quadratic slack
        Δx, Δλ, Δν, Δs = solve_kkt_quadratic_slack(x, s, λ, ν, μ_param)

        # Line search to find the step size
        t = line_search(x, s, λ, ν, Δx, Δλ, Δν, Δs)

        # Update primal and dual variables
        x += t * Δx
        λ += t * Δλ
        ν += t * Δν
        s += t * Δs

        # Update the barrier parameter
        μ_param *= 0.9  # Reduce the barrier parameter
    end

    return x, s, λ, ν
end
=#