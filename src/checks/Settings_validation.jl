"""
    validate_sequoia_settings!(settings::SEQUOIA_Settings)

Validates the fields of a `SEQUOIA_Settings` instance, applying default values where needed and issuing warnings.

# Arguments:
- `settings`: The `SEQUOIA_Settings` instance to validate.

# Modifies:
- Mutates `settings` by applying default values if required.
"""
function validate_sequoia_settings!(settings::SEQUOIA_Settings)
    validate_outer_method(settings.outer_method)
    validate_inner_solver(settings.inner_solver)
    validate_max_iter_outer(settings.max_iter_outer)
    validate_max_time_outer(settings.max_time_outer)
    validate_tolerance(settings.resid_tolerance, "resid_tolerance")
    validate_cost_min(settings.cost_min)
    
    # Specific validations for inner iterations and time based on convergence criteria
    settings.max_iter_inner = validate_max_iter_inner!(settings.max_iter_inner, settings.conv_crit)
    settings.max_time_inner = validate_max_time_inner!(settings.max_time_inner, settings.conv_crit)

    # Ensure cost_tolerance and cost_min are set if SEQUOIA and not solving a feasibility problem
    validate_cost_tolerance_and_min_cost!(settings)
end

"""
    validate_cost_tolerance_and_min_cost!(settings::SEQUOIA_Settings)

Ensures that cost tolerance and cost minimum are set appropriately when the outer method is SEQUOIA and feasibility is false.
"""
function validate_cost_tolerance_and_min_cost!(settings::SEQUOIA_Settings)
    if settings.outer_method == SEQUOIA && !settings.feasibility
        if settings.cost_tolerance === nothing
            @warn "Cost tolerance is `nothing` for non-feasibility problem with SEQUOIA. Defaulting to 1e-4."
            settings.cost_tolerance = 1e-4
        end
        if settings.cost_min === nothing
            @warn "Cost minimum is `nothing` for non-feasibility problem with SEQUOIA. Defaulting to -1e6."
            settings.cost_min = -1e6
        end
    end
end

"""
    validate_convergence_criterion!(settings::SEQUOIA_Settings)

Ensures that the settings for `max_iter_inner` and `max_time_inner` are appropriate for the selected convergence criterion.
"""
function validate_convergence_criterion!(settings::SEQUOIA_Settings)
    # Handle maximum iterations-based convergence criteria
    if settings.conv_crit in (MaxIterations, NormMaxIt, MaxItMaxTime, CombinedCrit, AdaptiveIterations)
        if settings.max_iter_inner === nothing
            @warn "Max iterations cannot be `nothing` for convergence criterion $(settings.conv_crit). Defaulting to 500."
            settings.max_iter_inner = 500
        end
    end

    # Handle time-based convergence criteria
    if settings.conv_crit in (MaxTime, MaxItMaxTime, NormMaxTime, CombinedCrit)
        if settings.max_time_inner === nothing
            @warn "Max time cannot be `nothing` for convergence criterion $(settings.conv_crit). Defaulting to 60 seconds."
            settings.max_time_inner = 60.0
        end
    end
end

"""
    validate_solver_params!(settings::SEQUOIA_Settings)

Checks if `solver_params` is valid based on the selected `outer_method`.
"""
function validate_solver_params!(settings::SEQUOIA_Settings)
    if settings.solver_params !== nothing
        if !all(x -> x isa Float64, settings.solver_params)
            error("Solver parameters must be a vector of Float64 values.")
        end
    end
end

"""
    validate_time_and_iterations!(settings::SEQUOIA_Settings)

Ensures that the provided time and iteration settings are positive and sensible.
"""
function validate_time_and_iterations!(settings::SEQUOIA_Settings)
    if settings.max_iter_outer <= 0
        error("Maximum outer iterations must be a positive integer.")
    end

    if settings.max_time_outer < 0
        error("Maximum outer time must be non-negative.")
    end

    if settings.max_iter_inner !== nothing && settings.max_iter_inner <= 0
        error("Maximum inner iterations must be a positive integer or `nothing`.")
    end

    if settings.max_time_inner !== nothing && settings.max_time_inner < 0
        error("Maximum inner time must be non-negative or `nothing`.")
    end

    if settings.resid_tolerance <= 0
        error("Residual tolerance must be a positive number.")
    end

    if settings.cost_tolerance !== nothing && settings.cost_tolerance <= 0
        error("Cost tolerance must be a positive number or `nothing`.")
    end
end

"""
    validate_inner_solver(inner_solver::InnerSolverEnum)

Validates that the inner solver is one of the supported options.
"""
function validate_inner_solver(inner_solver::InnerSolverEnum)
    if !(inner_solver in InnerSolverEnum)
        throw(ArgumentError("Invalid inner solver: $inner_solver. Valid solvers are: LBFGS, BFGS, Newton, GradientDescent, NelderMead."))
    end
end

"""
    validate_outer_method(outer_method::OuterMethodEnum)

Validates that the outer method is one of the supported options.
"""
function validate_outer_method(outer_method::OuterMethodEnum)
    if !(outer_method in OuterMethodEnum)
        throw(ArgumentError("Invalid outer method: $outer_method. Valid methods are: SEQUOIA, QPM, AugLag, IntPt."))
    end
end

"""
    validate_tolerance(tolerance::Real, name::String)

Validates that the tolerance is a positive real number.
"""
function validate_tolerance(tolerance::Real, name::String)
    if tolerance <= 0
        throw(ArgumentError("$name must be a positive number. Given: $tolerance"))
    end
end

"""
    validate_cost_min(cost_min::Union{Nothing, Real})

Validates that `cost_min` is reasonable and not too low.
"""
function validate_cost_min(cost_min::Union{Nothing, Real})
    if cost_min !== nothing && cost_min < -1e15
        throw(ArgumentError("cost_min is too low, suggesting potential issues. Given: $cost_min"))
    end
end

"""
    validate_max_iter_inner!(max_iter_inner::Union{Nothing, Int}, conv_crit::ConvCrit)

Validates or defaults `max_iter_inner` based on the convergence criterion.
"""
function validate_max_iter_inner!(max_iter_inner::Union{Nothing, Int}, conv_crit::ConvCrit)
    if conv_crit in (MaxIterations, NormMaxIt, MaxItMaxTime, CombinedCrit, AdaptiveIterations) && max_iter_inner === nothing
        @warn "max_iter_inner is required for convergence based on iteration count. Setting default to 500."
        return 500  # default value
    end
    return max_iter_inner
end

"""
    validate_max_time_inner!(max_time_inner::Union{Nothing, Real}, conv_crit::ConvCrit)

Validates or defaults `max_time_inner` based on the convergence criterion.
"""
function validate_max_time_inner!(max_time_inner::Union{Nothing, Real}, conv_crit::ConvCrit)
    if conv_crit in (MaxTime, MaxItMaxTime, NormMaxTime, CombinedCrit) && max_time_inner === nothing
        @warn "max_time_inner is required for convergence based on time limits. Setting default to 60 seconds."
        return 60.0  # default value in seconds
    end
    return max_time_inner
end
