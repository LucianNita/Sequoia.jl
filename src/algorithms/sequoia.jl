export sequoia_solve!

function sequoia_solve!(problem::SEQUOIA_pb, inner_solver, options, time, x, tk, iteration, inner_iterations)

    dk=problem.solver_settings.solver_params[1];
    gamma=problem.solver_settings.solver_params[2];
    beta=problem.solver_settings.solver_params[3];
    tk=problem.objective(x)#min(tk,problem.objective(x))

    x_tr=nothing;

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

        if dk<problem.solver_settings.cost_tolerance && rk <= problem.solver_settings.resid_tolerance         # Convergence check
            problem.solver_settings.store_trace && (x_tr=vcat(Optim.x_trace(resultl),Optim.x_trace(resultu)))
            step = SEQUOIA_Solution_step(iteration, dk, :first_order, time, inner_iterations, x, rk, problem.gradient(x), problem.constraints(x), [tk], x_tr)
            add_iterate!(problem.solution_history, step)  # Add step to history#

            return time, x, tk, iteration
        elseif dk<10^(-16) && rk > problem.solver_settings.resid_tolerance #problem.solver_settings.cost_tolerance
            problem.solver_settings.store_trace && (x_tr=vcat(Optim.x_trace(resultl),Optim.x_trace(resultu)))
            step = SEQUOIA_Solution_step(iteration, dk, :small_step, time, inner_iterations, x, rk, problem.gradient(x), problem.constraints(x), [tk], x_tr)
            add_iterate!(problem.solution_history, step)  # Add step to history#

            return time, x, tk, iteration
        elseif tk<problem.solver_settings.cost_min
            problem.solver_settings.store_trace && (x_tr=vcat(Optim.x_trace(resultl),Optim.x_trace(resultu)))
            step = SEQUOIA_Solution_step(iteration, dk, :unbounded, time, inner_iterations, x, rk, problem.gradient(x), problem.constraints(x), [tk], x_tr)
            add_iterate!(problem.solution_history, step)  # Add step to history#

            return time, x, tk, iteration
        end
        if problem.solver_settings.store_trace
            problem.solver_settings.store_trace && (x_tr=vcat(Optim.x_trace(resultl),Optim.x_trace(resultu)))
            step = SEQUOIA_Solution_step(iteration, dk, :unknown, resultl.time_run+resultu.time_run, resultl.iterations+resultu.iterations, x, rk, problem.gradient(x), problem.constraints(x), [tk], vcat(Optim.x_trace(resultl),Optim.x_trace(resultu)))
            add_iterate!(problem.solution_history, step)  # Add step to history
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
    if problem.solver_settings.store_trace
        problem.solution_history.iterates[end].solver_status=solver_status;
    else
        step = SEQUOIA_Solution_step(iteration, dk, solver_status, time, inner_iterations, x, rk, problem.gradient(x), problem.constraints(x), [tk])
        add_iterate!(problem.solution_history, step)  # Add step to history
    end

    return time, x, tu, iteration
end
