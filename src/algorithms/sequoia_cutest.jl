using SparseArrays

export sequoia_solve_cutest

function sequoia_solve_cutest(problem::CUTEstModel, inner_solver)
    # Extract initial parameters
    x = problem.meta.x0  # Initial guess
    tk=obj(problem,x)
    rtol = 1e-6

    t_tol=1e-4
    dk=2;
    gamma=2;
    beta=0.3;

    tu=tk;
    tl=tu-dk;

    # Penalty parameters
    max_iter_outer = 100
    max_iter_inner = 1000

    function opt(x,tk,rtol,max_iter_inner,inner_solver, problem)
        # Define the augmented objective and gradient functions
        obj_aug_fn = x -> r(x, tk, problem)
        grad_aug_fn! = (g, x) -> r_gradient!(g, x, tk, problem)
    
        # Optim.jl options for inner solve
        options = Optim.Options(g_tol=rtol, iterations=max_iter_inner, store_trace=true, extended_trace=true, show_trace=false)
    
        # Solve the unconstrained subproblem
        result = Optim.optimize(obj_aug_fn, grad_aug_fn!, x, inner_solver, options)
        return result
    end

    # Initialize iteration variables
    iteration = 1
    time_elapsed = 0.0
    solution_history = []
    rk=r(x,tk,problem)

    while iteration < max_iter_outer
        
        resultu=opt(x,tu,rtol,max_iter_inner,inner_solver,problem)
        resultl=opt(x,tl,rtol,max_iter_inner,inner_solver,problem)

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
            inner_comp_time = result.time_run  # Computation time for this inner solve

            # Compute constraint violation
            constraint_violation = exact_augmented_constraint_violation(x,tk,problem)
            res_val = result.minimum;

            # Log the step
            step = Dict(
                "iteration" => iteration,
                "objective" => result.minimum,
                "x" => x,
                "constraint_violation" => constraint_violation,
                "time" => inner_comp_time
            )
            push!(solution_history, step)
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
        #time_elapsed += inner_comp_time
    end

    return solution_history
end
