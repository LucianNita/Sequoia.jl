import Optim
using LinearAlgebra

export solve!

"""
    solve!(problem::SEQUOIA_pb)

Solve a constrained optimization problem using SEQUOIA.

# Arguments
- `problem::SEQUOIA_pb`: The optimization problem to solve, including settings, initial guess, and constraints.

# Workflow
1. Initializes inner solver based on `problem.solver_settings.inner_solver`.
2. Validates solver parameters for the selected outer method (`:QPM`, `:AugLag`, `:IntPt`, `:SEQUOIA`).
3. If `problem.solver_settings.feasibility` is `true`, runs a feasibility solve.
4. Delegates the main optimization task to the appropriate solver (`qpm_solve!`, `alm_solve!`, `ipm_solve!`, `sequoia_solve!`).

# Returns
- Updates the `problem.solution_history` and modifies `problem.x0` with the final solution.

# Notes
- Supports multiple outer methods and solvers with customizable settings.
- Throws an error if solver parameters are mismatched or unsupported methods are used.

"""
function solve!(problem::SEQUOIA_pb)
    # Inner solver from SEQUOIA_Settings
    inner_solver = choose_inner_solver(problem.solver_settings.inner_solver)
    options = set_options(problem.solver_settings)

    x = problem.x0
    iteration = 0;
    inner_iterations = 0;
    time = 0.0 # Initialization of computational time
    problem.solution_history = SEQUOIA_History()  # Initialize the solution history
    previous_fval = Inf  # Store the previous objective function value

    if problem.solver_settings.feasibility
        time, x, previous_fval, inner_iterations = feasibility_solve!(problem,inner_solver,options, time, x, previous_fval, inner_iterations)
    end

    iteration = 1  # Initialize iteration counter
    if problem.solver_settings.outer_method == :QPM
        
        if problem.solver_settings.solver_params === nothing
            problem.solver_settings.solver_params=[1.0,10.0,10.0,0.0];
        elseif length(problem.solver_settings.solver_params) != 4
            throw(ArgumentError("Expected 4 parameters for QPM solver; received $(length(problem.solver_settings.solver_params))."))
        end
        time, x, previous_fval, iteration, inner_iterations = qpm_solve!(problem, inner_solver, options, time, x, previous_fval, iteration, inner_iterations)

    elseif problem.solver_settings.outer_method == :AugLag
        
        if problem.cutest_nlp === nothing
            clen=length(problem.eqcon)+length(problem.ineqcon);
        else
            clen=length(problem.cutest_nlp.meta.jfix)+length(problem.cutest_nlp.meta.jlow)+length(problem.cutest_nlp.meta.jupp)+2*length(problem.cutest_nlp.meta.jrng)+length(problem.cutest_nlp.meta.ifix)+length(problem.cutest_nlp.meta.ilow)+length(problem.cutest_nlp.meta.iupp)+2*length(problem.cutest_nlp.meta.irng)
        end
        if problem.solver_settings.solver_params === nothing
            problem.solver_settings.solver_params=vcat([1.0,10.0,10.0,0.0],zeros(clen));
        elseif length(problem.solver_settings.solver_params) != 4+clen
            throw(ArgumentError("Number of solver parameters is incompatible with the solver. Current solver chosen $(problem.solver_settings.outer_method). Number of expected parameters $(4+length(problem.eq_indices)+length(problem.ineq_indices)), got $(length(problem.solver_settings.solver_params)). Please modify the settings by either choosing a different solver, providing an appropriate number of parameters, or leaving the optional field free."))
        end
        time, x, previous_fval, iteration, inner_iterations = alm_solve!(problem, inner_solver, options, time, x, previous_fval, iteration, inner_iterations)
        
    elseif problem.solver_settings.outer_method == :IntPt

        if problem.cutest_nlp === nothing
            clen=length(problem.eqcon)+length(problem.ineqcon);
        else
            clen=length(problem.cutest_nlp.meta.jfix)+length(problem.cutest_nlp.meta.jlow)+length(problem.cutest_nlp.meta.jupp)+2*length(problem.cutest_nlp.meta.jrng)+length(problem.cutest_nlp.meta.ifix)+length(problem.cutest_nlp.meta.ilow)+length(problem.cutest_nlp.meta.iupp)+2*length(problem.cutest_nlp.meta.irng)
        end
        if problem.solver_settings.solver_params === nothing
            problem.solver_settings.solver_params=vcat([1.0,0.5,0.1,0.0],zeros(clen));
        elseif length(problem.solver_settings.solver_params) != 4+clen
            throw(ArgumentError("Number of solver parameters is incompatible with the solver. Current solver chosen $(problem.solver_settings.outer_method). Number of expected parameters $(4+length(problem.eq_indices)+length(problem.ineq_indices)), got $(length(problem.solver_settings.solver_params)). Please modify the settings by either choosing a different solver, providing an appropriate number of parameters, or leaving the optional field free."))
        end
        if problem.solver_settings.cost_tolerance === nothing
            problem.solver_settings.cost_tolerance = 10^(-6);
        end
        time, x, previous_fval, iteration, inner_iterations = ipm_solve!(problem, inner_solver, options, time, x, previous_fval, iteration, inner_iterations)

    elseif problem.solver_settings.outer_method == :SEQUOIA

        if problem.solver_settings.solver_params === nothing
            problem.solver_settings.solver_params=[2.0,2.0,0.3];
        elseif length(problem.solver_settings.solver_params) != 3
            throw(ArgumentError("Expected 3 parameters for SEQUOIA solver; received $(length(problem.solver_settings.solver_params))."))
        end
        time, x, previous_fval, iteration, inner_iterations = sequoia_solve!(problem, inner_solver, options, time, x, previous_fval, iteration, inner_iterations)


    else
        error("The provided outer_method is not supported. Only QPM, AugLag, IntPt and SEQUOIA are implemented and tested.")
    end
end

"""
    choose_inner_solver(inner_solver::Symbol)

Select the appropriate `Optim.jl` solver for the given inner solver symbol.

# Arguments
- `inner_solver::Symbol`: The name of the desired inner solver (e.g., `:LBFGS`).

# Returns
- An instance of the corresponding `Optim.jl` solver.

# Supported Solvers
- `:LBFGS`
- `:BFGS`
- `:Newton`
- `:GradientDescent`
- `:NelderMead`

# Notes
- Throws an error if the solver name is invalid.
"""
function choose_inner_solver(inner_solver::Symbol)
    if inner_solver==:LBFGS
        return Optim.LBFGS()
    elseif inner_solver==:BFGS
        return Optim.BFGS()
    elseif inner_solver==:Newton
        return Optim.Newton()
    elseif inner_solver==:GradientDescent
        return Optim.GradientDescent()
    elseif inner_solver==:NelderMead
        return Optim.NelderMead()
    else
        error("Unknown inner solver: $inner_solver. Make sure you use one of the accepted and tested solvers.")
    end
end

"""
    set_options(settings::SEQUOIA_Settings)

Configure options for the `Optim.jl` solver based on SEQUOIA settings.

# Arguments
- `settings::SEQUOIA_Settings`: Optimization settings including convergence criteria, tolerance, and trace options.

# Returns
- An instance of `Optim.Options` configured for the specified criteria.

# Notes
- Supports `:GradientNorm`, `:MaxIterations`, `:MaxTime`, and `:CombinedCrit` as convergence criteria.
- Throws an error if `settings.conv_crit` is invalid.
"""
function set_options(settings::SEQUOIA_Settings)        # Set Optim options
    if settings.conv_crit==:GradientNorm
        options = Optim.Options(g_tol=settings.resid_tolerance, store_trace=settings.store_trace, extended_trace=settings.store_trace, show_trace=false)
    elseif settings.conv_crit==:MaxIterations
        options = Optim.Options(g_tol=settings.resid_tolerance, iterations=settings.max_iter_inner, store_trace=settings.store_trace, extended_trace=settings.store_tracetrue, show_trace=false)
    elseif settings.conv_crit==:MaxTime
        options = Optim.Options(g_tol=settings.resid_tolerance, time_limit=settings.max_time_inner, store_trace=settings.store_trace, extended_trace=settings.store_trace, show_trace=false)
    elseif settings.conv_crit==:CombinedCrit
        options = Optim.Options(g_tol=settings.resid_tolerance, iterations=settings.max_iter_inner, time_limit=settings.max_time_inner, store_trace=settings.store_trace, extended_trace=settings.store_trace, show_trace=false)
    else
        throw(ArgumentError("Invalid convergence criterion: $conv_crit. Valid criteria are: $(join(valid_convergence_criterias, ", "))."))
    end

    return options
end