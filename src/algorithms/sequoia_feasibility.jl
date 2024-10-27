using Optim
using SparseArrays
using LinearAlgebra
using NLPModels
using CUTEst
using ForwardDiff

function sequoia_feasibility_cutest(problem::CUTEstModel, inner_solver)
    # Extract initial parameters
    x = problem.meta.x0  # Initial guess
    max_iter_inner = 1000

    # Function to calculate the augmented objective
    function residual(x)
        obj_val = 0.0;
        constraint_val = cons(problem, x)
        eq_penalty_term = 0.0
        lineq_penalty_term = 0.0
        uineq_penalty_term = 0.0
        rineq_penalty_term = 0.0

        var_fix = 0.0;
        var_low = 0.0;
        var_upp = 0.0;
        var_rng = 0.0;

        # Handle equality constraints (penalty applied if violated)
        for i in eachindex(problem.meta.jfix)
            eq_penalty_term += 0.5 * (constraint_val[problem.meta.jfix[i]]-problem.meta.lcon[problem.meta.jfix[i]])^2
        end

        # Handle inequality constraints (penalty applied if outside bounds)
        for i in eachindex(problem.meta.jlow)
            if constraint_val[problem.meta.jlow[i]] < problem.meta.lcon[problem.meta.jlow[i]]
                lineq_penalty_term += 0.5 * (problem.meta.lcon[problem.meta.jlow[i]]-constraint_val[problem.meta.jlow[i]])^2
            end
        end

        for i in eachindex(problem.meta.jupp)
            if constraint_val[problem.meta.jupp[i]] > problem.meta.ucon[problem.meta.jupp[i]]
                uineq_penalty_term += 0.5 * (constraint_val[problem.meta.jupp[i]]-problem.meta.ucon[problem.meta.jupp[i]])^2
            end
        end

        for i in eachindex(problem.meta.jrng)
            if constraint_val[problem.meta.jrng[i]] > problem.meta.ucon[problem.meta.jrng[i]]
                rineq_penalty_term += 0.5 * (constraint_val[problem.meta.jrng[i]]-problem.meta.ucon[problem.meta.jrng[i]])^2
            end
            if constraint_val[problem.meta.jrng[i]] < problem.meta.lcon[problem.meta.jrng[i]]
                rineq_penalty_term += 0.5 * (problem.meta.lcon[problem.meta.jrng[i]]-constraint_val[problem.meta.jrng[i]])^2
            end
        end

        for i in eachindex(problem.meta.ifix)
            var_fix += 0.5 * (x[problem.meta.ifix[i]]-problem.meta.lvar[i])^2
        end

        # Handle inequality constraints (penalty applied if outside bounds)
        for i in eachindex(problem.meta.ilow)
            if x[problem.meta.ilow[i]] < problem.meta.lvar[problem.meta.ilow[i]]
                var_low += 0.5 * (problem.meta.lvar[problem.meta.ilow[i]]-x[problem.meta.ilow[i]])^2
            end
        end

        for i in eachindex(problem.meta.iupp)
            if x[problem.meta.iupp[i]] > problem.meta.uvar[problem.meta.iupp[i]]
                var_upp += 0.5 * (x[problem.meta.iupp[i]]-problem.meta.uvar[problem.meta.iupp[i]])^2
            end
        end

        for i in eachindex(problem.meta.irng)
            if x[problem.meta.irng[i]] < problem.meta.lvar[problem.meta.irng[i]]
                var_rng += 0.5 * (problem.meta.lvar[problem.meta.irng[i]]-x[problem.meta.irng[i]])^2
            end
            if x[problem.meta.irng[i]] > problem.meta.uvar[problem.meta.irng[i]]
                var_rng += 0.5 * (x[problem.meta.irng[i]]-problem.meta.uvar[problem.meta.irng[i]])^2
            end
        end


        return obj_val + eq_penalty_term + lineq_penalty_term + uineq_penalty_term + rineq_penalty_term + var_fix + var_low + var_upp + var_rng 
    end

    function augmented_gradient!(grad_storage, x)

        # Get constraint values and the Jacobian
        constraint_val = cons(problem, x)
        jacobian = jac(problem, x)  # This will be a sparse matrix

        # Initialize penalty gradients
        grad_obj = zeros(length(x))
        grad_eq_penalty = zeros(length(x))
        grad_ineq_penalty = zeros(length(x))
        grad_var_penalty = zeros(length(x))

        # Equality constraints
        for i in eachindex(problem.meta.jfix)
            idx = problem.meta.jfix[i]
            violation = constraint_val[idx] - problem.meta.lcon[idx]
            grad_eq_penalty += jacobian[idx, :] * violation
        end

        # Inequality constraints: lower bounds
        for i in eachindex(problem.meta.jlow)
            idx = problem.meta.jlow[i]
            if constraint_val[idx] < problem.meta.lcon[idx]
                violation = constraint_val[idx] - problem.meta.lcon[idx]
                grad_ineq_penalty += jacobian[idx, :] * violation
            end
        end

        # Inequality constraints: upper bounds
        for i in eachindex(problem.meta.jupp)
            idx = problem.meta.jupp[i]
            if constraint_val[idx] > problem.meta.ucon[idx]
                violation = constraint_val[idx] - problem.meta.ucon[idx]
                grad_ineq_penalty += jacobian[idx, :] * violation 
            end
        end

        # Inequality constraints: range constraints
        for i in eachindex(problem.meta.jrng)
            idx = problem.meta.jrng[i]
            if constraint_val[idx] > problem.meta.ucon[idx]
                violation = constraint_val[idx] - problem.meta.ucon[idx]
                grad_ineq_penalty += jacobian[idx, :] * violation 
            end
            if constraint_val[idx] < problem.meta.lcon[idx]
                violation = constraint_val[idx] - problem.meta.lcon[idx]
                grad_ineq_penalty += jacobian[idx, :] * violation 
            end
        end

        # Variable penalties: fixed, lower, upper, and range
        for i in eachindex(problem.meta.ifix)
            idx = problem.meta.ifix[i]
            violation = x[idx] - problem.meta.lvar[idx]
            grad_var_penalty[idx] += violation
        end

        for i in eachindex(problem.meta.ilow)
            idx = problem.meta.ilow[i]
            if x[idx] < problem.meta.lvar[idx]
                violation = x[idx] - problem.meta.lvar[idx]
                grad_var_penalty[idx] += violation
            end
        end

        for i in eachindex(problem.meta.iupp)
            idx = problem.meta.iupp[i]
            if x[idx] > problem.meta.uvar[idx]
                violation = x[idx] - problem.meta.uvar[idx]
                grad_var_penalty[idx] += violation
            end
        end

        for i in eachindex(problem.meta.irng)
            idx = problem.meta.irng[i]
            if x[idx] > problem.meta.uvar[idx]
                violation = x[idx] - problem.meta.uvar[idx]
                grad_var_penalty[idx] += violation
            end
            if x[idx] < problem.meta.lvar[idx]
                violation = x[idx] - problem.meta.lvar[idx]
                grad_var_penalty[idx] += violation
            end
        end

        # Combine all gradients: objective + penalty terms
        grad_storage .= grad_obj .+ grad_eq_penalty .+ grad_ineq_penalty .+ grad_var_penalty

    end
    

    # Function to compute the total constraint violation (used for adaptive penalty updates)
    function compute_constraint_violation(x)
        constraint_val = cons(problem, x)
        eq_violation = norm(constraint_val[problem.meta.jfix]-problem.meta.lcon[problem.meta.jfix])  # Equality violation
        lineq_violation = norm(max.(0.0, problem.meta.lcon[problem.meta.jlow]-constraint_val[problem.meta.jlow]))  # Inequality violation
        uineq_violation = norm(max.(0.0, constraint_val[problem.meta.jupp]-problem.meta.ucon[problem.meta.jupp]))  # Inequality violation
        rineq_violation = norm(max.(0.0, problem.meta.lcon[problem.meta.jrng]-constraint_val[problem.meta.jrng], constraint_val[problem.meta.jrng]-problem.meta.ucon[problem.meta.jrng]))

        var_fix_violation = norm(x[problem.meta.ifix]-problem.meta.lvar[problem.meta.ifix]);
        var_low_violation = norm(max.(0.0, problem.meta.lvar[problem.meta.ilow]-x[problem.meta.ilow]))  # Inequality violation
        var_upp_violation = norm(max.(0.0, x[problem.meta.iupp]-problem.meta.uvar[problem.meta.iupp]))
        var_rng_violation = norm(max.(0.0, problem.meta.lvar[problem.meta.irng]-x[problem.meta.irng], x[problem.meta.irng]-problem.meta.uvar[problem.meta.irng]))

        return eq_violation + lineq_violation + uineq_violation + rineq_violation + var_fix_violation + var_low_violation + var_upp_violation + var_rng_violation
    end


    # Initialize iteration variables
    iteration = 1
    time_elapsed = 0.0
    solution_history = []
    rk=residual(x)

        obj_aug_fn = x -> residual(x)
        grad_aug_fn! = (g, x) -> augmented_gradient!(g, x)
    
        # Optim.jl options for inner solve
        options = Optim.Options(g_tol=rtol, iterations=max_iter_inner, store_trace=true, extended_trace=true, show_trace=false)
    
        # Solve the unconstrained subproblem
        result = Optim.optimize(obj_aug_fn, grad_aug_fn!, x, inner_solver, options)



        # Extract the solution
        x = result.minimizer
        inner_comp_time = result.time_run  # Computation time for this inner solve

        # Compute constraint violation
        constraint_violation = compute_constraint_violation(x)

        # Log the step
        step = Dict(
            "iteration" => iteration,
            "objective" => result.minimum,
            "x" => x,
            "constraint_violation" => constraint_violation,
            "time" => inner_comp_time
        )
        push!(solution_history, step)

    return solution_history
end



problem=CUTEstModel("BT1");
sol_hist=sequoia_solve_cutest(problem,Optim.LBFGS())
finalize(problem)