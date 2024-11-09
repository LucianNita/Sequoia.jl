export validate_sequoia_settings!

"""
    validate_sequoia_settings!(settings::SEQUOIA_Settings)

Validates the fields of a `SEQUOIA_Settings` instance, applying default values where needed and issuing warnings.

# Arguments:
- `settings`: The `SEQUOIA_Settings` instance to validate.

# Modifies:
- Mutates `settings` by applying default values if required.
"""
function validate_sequoia_settings!(settings::SEQUOIA_Settings)
    # Validate core fields
    validate_outer_method(settings.outer_method)
    validate_inner_solver(settings.inner_solver)
    validate_convergence_criteria(settings.conv_crit)
    
    # Validate numeric parameters
    validate_numeric!(settings)
    
    # Apply defaults for optional settings
    apply_defaults!(settings)
end

"""
    validate_inner_solver(inner_solver::Symbol)

Validates that the inner solver is one of the supported options. Throws an ArgumentError if the inner solver is invalid.
"""
function validate_inner_solver(inner_solver::Symbol)
    if !(inner_solver in inner_solvers)
        throw(ArgumentError("Invalid inner solver: $inner_solver. Valid solvers are: $(join(inner_solvers, ", "))."))
    end
end

"""
    validate_outer_method(outer_method::Symbol)

Validates that the outer method is one of the supported options. Throws an ArgumentError if the outer method is invalid.
"""
function validate_outer_method(outer_method::Symbol)
    if !(outer_method in outer_methods)
        throw(ArgumentError("Invalid outer method: $outer_method. Valid methods are: $(join(outer_methods, ", "))."))
    end
end
"""
    validate_convergence_criteria(conv_crit::Symbol)

Validates that the convergence criterion is one of the supported options. 
Throws an ArgumentError if the convergence criterion is invalid.
"""
function validate_convergence_criteria(conv_crit::Symbol)
    valid_convergence_criterias = [
        :GradientNorm, :MaxIterations, :MaxTime, :ConstraintResidual, 
        :NormMaxIt, :MaxItMaxTime, :NormMaxTime, :CombinedCrit, :AdaptiveIterations
    ]
    if !(conv_crit in valid_convergence_criterias)
        throw(ArgumentError("Invalid convergence criterion: $conv_crit. Valid options are: $(join(convergence_criteria, ", "))."))
    end
end

"""
    validate_numeric!(settings::SEQUOIA_Settings)

Validates numeric parameters such as iteration counts, times, and tolerances.
"""
function validate_numeric!(settings::SEQUOIA_Settings)
    # Ensure positive iterations and times
    if settings.max_iter_outer <= 0
        throw(ArgumentError("Maximum outer iterations must be a positive integer."))
    end
    if settings.max_time_outer <= 0
        throw(ArgumentError("Maximum outer time must be non-negative."))
    end
    if settings.resid_tolerance <= 0
        throw(ArgumentError("Residual tolerance must be a positive number."))
    end

    # Validate optional fields
    if settings.max_iter_inner !== nothing && settings.max_iter_inner <= 0
        throw(ArgumentError("Maximum inner iterations must be a positive integer or `nothing`."))
    end
    if settings.max_time_inner !== nothing && settings.max_time_inner <= 0
        throw(ArgumentError("Maximum inner time must be positive or `nothing`."))
    end
    if settings.cost_tolerance !== nothing && settings.cost_tolerance <= 0
        throw(ArgumentError("Cost tolerance must be a positive number or `nothing`."))
    end
    if settings.cost_min !== nothing && settings.cost_min < -1e15
        throw(ArgumentError("Cost minimum is too low, suggesting potential issues."))
    end
    if settings.step_min !== nothing && (settings.cost_min < 1e-20 || settings.cost_min > 1e-4)
        throw(ArgumentError("Minimum step size must be in the range [1e-20, 1e-4]. Provided: $(settings.step_min)."))
    end
end

"""
    apply_defaults!(settings::SEQUOIA_Settings)

Applies default values for optional parameters if not provided.
"""
function apply_defaults!(settings::SEQUOIA_Settings)
    # Defaults for SEQUOIA-specific settings
    if settings.outer_method == :SEQUOIA
        if settings.cost_tolerance === nothing
            @warn "Cost tolerance is `nothing`. Defaulting to 1e-4."
            settings.cost_tolerance = 1e-4
        end
        if settings.cost_min === nothing
            @warn "Cost minimum is `nothing`. Defaulting to -1e6."
            settings.cost_min = -1e6
        end
    end

    # Defaults for inner solver settings
    if settings.max_iter_inner === nothing && settings.conv_crit in [:MaxIterations, :CombinedCrit]
        @warn "max_iter_inner is missing for iteration-based convergence. Defaulting to 500."
        settings.max_iter_inner = 500
    end
    if settings.max_time_inner === nothing && settings.conv_crit in [:MaxTime, :CombinedCrit]
        @warn "max_time_inner is missing for time-based convergence. Defaulting to 60 seconds."
        settings.max_time_inner = 60.0
    end
end
