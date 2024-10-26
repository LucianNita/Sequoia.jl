import Optim
using LinearAlgebra

export solve!

function solve!(problem::SEQUOIA_pb)
    validate_pb(problem)

    # Inner solver from SEQUOIA_Settings
    inner_solver = choose_inner_solver(problem.solver_settings.inner_solver)


    if problem.solver_settings.outer_method == :QPM
        if problem.solver_settings.solver_params === nothing
            problem.solver_settings.solver_params=[1.0,10.0,10.0,0.0];
        elseif length(problem.solver_settings.solver_params) != 4
            throw(ArgumentError("Number of solver parameters is incompatible with the solver. Current solver chosen $(problem.solver_settings.outer_method). Number of expected parameters 4, got $(length(problem.solver_settings.solver_params)). Please modify the settings by either choosing a different solver, providing an appropriate number of parameters, or leaving the optional field free."))
        end

        #  Call the QPM algorithm with extracted data, returns solution history
        problem.solution_history = qpm_solve(problem,inner_solver)

    elseif problem.solver_settings.outer_method == :AugLag
        if problem.solver_settings.solver_params === nothing
            problem.solver_settings.solver_params=vcat([1.0,10.0,10.0,0.0],zeros(length(problem.eq_indices)+length(problem.ineq_indices)));
        elseif length(problem.solver_settings.solver_params) != 4+length(problem.eq_indices)+length(problem.ineq_indices)
            throw(ArgumentError("Number of solver parameters is incompatible with the solver. Current solver chosen $(problem.solver_settings.outer_method). Number of expected parameters $(4+length(problem.eq_indices)+length(problem.ineq_indices)), got $(length(problem.solver_settings.solver_params)). Please modify the settings by either choosing a different solver, providing an appropriate number of parameters, or leaving the optional field free."))
        end

        # Call the ALM algorithm with extracted data, returns solution history
        problem.solution_history = alm_solve(problem, inner_solver)
    else
        error("The provided outer_method is not supported. Only QPM, AugLag are implemented.")
    end
end



# Helper function to map SEQUOIA_pb inner solvers to Optim.jl solvers
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
