
function sequoia_solve(problem::SEQUOIA_pb, inner_solver, options)

    x=problem.x0;  # Initial guess
    tk=problem.objective(x)
    rtol=problem.solver_settings.resid_tolerance;

    t_tol=problem.solver_settings.cost_tolerance;
    
    dk=problem.solver_settings.solver_params[1];
    gamma=problem.solver_settings.solver_params[2];
    beta=problem.solver_settings.solver_params[3];

    tu=tk;
    tl=tu-dk;

    # Initialize iteration variables
    iteration = 1
    time = 0.0
    solution_history = SEQUOIA_History();
    if problem.cutest_nlp === nothing
        rk=r(x,tk,problem)
    else
        rk=r(x,tk,problem.cutest_nlp)
    end
    
    while iteration < problem.solver_settings.max_iter_outer && time < problem.solver_settings.max_time_outer
        
        if problem.cutest_nlp === nothing
            obj_aug_fnu = x -> r(x, tu, problem)
            grad_aug_fnu! = (g, x) -> r_gradient!(g, x, tu, problem)
            obj_aug_fnl = x -> r(x, tl, problem)
            grad_aug_fnl! = (g, x) -> r_gradient!(g, x, tl, problem)
        else
            obj_aug_fnu = x -> r(x, tu, problem.cutest_nlp)
            grad_aug_fnu! = (g, x) -> r_gradient!(g, x, tu, problem.cutest_nlp)
            obj_aug_fnl = x -> r(x, tl, problem.cutest_nlp)
            grad_aug_fnl! = (g, x) -> r_gradient!(g, x, tl, problem.cutest_nlp)
        end

        resultu = Optim.optimize(obj_aug_fnu, grad_aug_fnu!, x, inner_solver, options)
        resultl = Optim.optimize(obj_aug_fnl, grad_aug_fnl!, x, inner_solver, options)


        if @isdefined resultk
            candidates = [(resultl, resultl.minimum, tl), (resultk, rk, tk), (resultu, resultu.minimum, tu)]
        else 
            candidates = [(resultl, resultl.minimum, tl), (resultu, resultu.minimum, tu)]
        end

        below_tolerance = [(x, r, t) for (x, r, t) in candidates if r <= rtol]

        if !isempty(below_tolerance)
            # Choose the one with the smallest t among those below tolerance
            idx=findmin([(t, x, r, t) for (x, r, t) in below_tolerance])[2]
            (results,rs,ts)=below_tolerance[idx]
        else
            # Choose the one with the smallest r if none are below tolerance
            idx=findmin([(r, x, r, t) for (x, r, t) in candidates])[2]
            (results,rs,ts)=candidates[idx]
        end

        if (rs<=rtol  && ts < tk) || rs<rk 
            result=results;
            rk=rs;
            tk=ts;
            dk*=gamma;
        else
            dk*=beta
        end

        if rk>rtol
            tu=tk+dk
        else
            tu=tk
        end
        tl=tk-dk;

        if @isdefined result
            # Extract the solution
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
            solver_status = Optim.converged(result) ? :success : :not_converged  # Solver status
            inner_comp_time = result.time_run  # Computation time
            num_inner_iterations = result.iterations  # Number of inner iterations
            x_tr = Optim.x_trace(result)  # This returns the history of iterates
            conv=constraint_violation;
            # Create a SEQUOIA_Solution_step and save it to history
            step = SEQUOIA_Solution_step(iteration, conv, solver_status, inner_comp_time, num_inner_iterations, x, fval, gval, cval, [resultl.minimum,rk,result.minimum,tl,tk,tu,dk], x_tr)
            add_iterate!(solution_history, step)  # Add step to history
        end

        # Convergence check
        if rk <= rtol 
            tu=tk;
        else 
            tu=tk+dk;
        end

        tl=tk-dk

        if tu-tl<t_tol
            println("Converged after $(iteration) iterations.")
            return solution_history
        end

        # Increment iteration counter
        iteration += 1
        time+=resultl.time_run+resultu.time_run;
    end

    return solution_history
end
