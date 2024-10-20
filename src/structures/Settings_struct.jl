export SEQUOIA_Settings

@enum InnerSolverEnum LBFGS BFGS Newton GradientDescent NelderMead
@enum OuterMethodEnum SEQUOIA QPM AugLag IntPt

# Define different convergence criteria methods
@enum ConvCrit begin
    GradientNorm        # Convergence based on gradient norm
    MaxIterations       # Convergence based on maximum number of iterations
    MaxTime             # Convergence based on maximum computational time
    ConstraintResidual  # Convergence based on constraint residual - relevant for feasibility
    NormMaxIt           # Combined criterion: either gradient norm below tolerance or number of iterations exceeded
    MaxItMaxTime        # Combined criterion: either number of iterations exceeded or maximum time exceeded
    NormMaxTime         # Combined criterion: either gradient norm below tolerance or maximum time exceeded
    CombinedCrit        # Combined criterion: either gradient norm below tolerance or number of iterations exceeded or maximum time exceeded
    AdaptiveIterations  # Adaptive number of inner iterations based on outer iteration count
end

"""
    SEQUOIA_Settings

The `SEQUOIA_Settings` struct stores the key settings required for solving optimization problems using the SEQUOIA method. It allows configuration of inner and outer solvers, convergence criteria, and various numerical tolerances.

# Fields

- `outer_method::OuterMethodEnum`: Specifies the outer optimization method. Valid options include:
    - `:SEQUOIA`, `:QPM`, `:AugLag`, `:IntPt` (symbols or equivalent strings are accepted).
  
- `inner_solver::InnerSolverEnum`: Specifies the inner solver used to solve unconstrained subproblems. Valid options include:
    - `:LBFGS`, `:BFGS`, `:Newton`, `:GradientDescent`, `:NelderMead` (symbols or equivalent strings are accepted).

- `feasibility::Bool`: Determines if the solver is solving a feasibility problem (i.e., ensuring constraints are satisfied) or optimizing an objective function.

- `resid_tolerance::Real`: Residual tolerance for constraints. The solver considers the constraints satisfied when the residual is below this value. Must be a positive number.

- `max_iter_outer::Int`: Maximum number of iterations allowed for the outer solver. Must be a positive integer.

- `max_time_outer::Real`: Maximum computational time (in seconds) for the entire optimization process. Set to `Inf` for no time limit. Must be non-negative.

- `conv_crit::ConvCrit`: Specifies the convergence criterion for the inner solver. Valid options include:
    - `:GradientNorm`, `:MaxIterations`, `:MaxTime`, `:ConstraintResidual`, etc. (symbols or equivalent strings are accepted).

- `max_iter_inner::Union{Nothing, Int}`: Maximum number of iterations for the inner solver. If `nothing`, the default behavior of the inner solver is used.

- `max_time_inner::Union{Nothing, Real}`: Maximum time (in seconds) for each inner solve call. Set to `nothing` for no limit or a positive real number to impose a time limit.

- `cost_tolerance::Union{Nothing, Real}`: Desired optimality gap. The solver stops when the difference between the current solution and the optimal cost is below this threshold. Must be positive if specified, or `nothing` for no specific cost tolerance.

- `cost_min::Union{Nothing, Real}`: Minimum allowed cost value. This helps detect unbounded problems. Can be set to `nothing` if not needed.

- `solver_params::Union{Nothing, Vector{Float64}}`: Optional solver-related parameters, such as step sizes, penalty parameters, or Lagrange multipliers. Defaults to `nothing` if not specified.

# Constructors

## Full Constructor
Allows complete control over all fields of the `SEQUOIA_Settings` struct.

```julia
SEQUOIA_Settings(
    outer_method::Union{Symbol, String},
    inner_solver::Union{Symbol, String},
    feasibility::Bool,
    resid_tolerance::Real,
    max_iter_outer::Int,
    max_time_outer::Real,
    conv_crit::Union{Symbol, String},
    max_iter_inner::Union{Nothing, Int},
    max_time_inner::Union{Nothing, Real},
    cost_tolerance::Union{Nothing, Real},
    cost_min::Union{Nothing, Real},
    solver_params::Union{Nothing, Vector{Float64}} = nothing
)

# Full constructor example
settings_full = SEQUOIA_Settings(
    :QPM,
    :LBFGS,
    false,
    1e-8,
    1000,
    3600.0,
    :GradientNorm,
    nothing,
    nothing,
    1e-4,
    -1e6,
    [1.0, 0.5]
)

# Minimal constructor example
settings_min = SEQUOIA_Settings(
    :QPM,
    :LBFGS,
    false,
    1e-6,
    1000,
    3600.0
)
"""
mutable struct SEQUOIA_Settings
    outer_method::OuterMethodEnum                   # Outer optimization method (SEQUOIA, QPM, AugLag, IntPt).
    inner_solver::InnerSolverEnum                   # The inner solver used for unconstrained problems.
    feasibility::Bool                               # Solve feasibility problem (true) or account for an objective (false)?
    
    resid_tolerance::Real                           # Residual tolerance for constraints.
    max_iter_outer::Int                             # Maximum number of outer iterations allowed.
    max_time_outer::Real                            # Maximum total computational time (in seconds).

    conv_crit::ConvCrit                             # Convergence criterion for the inner solve.
    max_iter_inner::Union{Nothing, Int}             # Maximum number of inner iterations allowed.
    max_time_inner::Union{Nothing, Real}            # Maximum computational time for each inner solve call (in seconds).

    cost_tolerance::Union{Nothing, Real}            # Desired optimality gap.
    cost_min::Union{Nothing, Real}                  # Minimum cost - useful for spotting possible unbounded problems.

    solver_params::Union{Nothing, Vector{Float64}}  # Optional solver-related parameters.

    # Full constructor with symbol/string support
    function SEQUOIA_Settings(outer_method::Union{Symbol, String}, inner_solver::Union{Symbol, String}, feasibility::Bool, 
                              resid_tolerance::Real, max_iter_outer::Int, max_time_outer::Real, 
                              conv_crit::Union{Symbol, String}, max_iter_inner::Union{Nothing, Int}, max_time_inner::Union{Nothing, Real}, 
                              cost_tolerance::Union{Nothing, Real}, cost_min::Union{Nothing, Real}, 
                              solver_params::Union{Nothing, Vector{Float64}} = nothing)
        return new(parse_outer_method(outer_method), parse_inner_solver(inner_solver), feasibility, 
                   resid_tolerance, max_iter_outer, max_time_outer, 
                   parse_convergence_criterion(conv_crit), max_iter_inner, max_time_inner, 
                   cost_tolerance, cost_min, solver_params)
    end

    # Minimal constructor with symbol/string support
    function SEQUOIA_Settings(outer_method::Union{Symbol, String}, inner_solver::Union{Symbol, String}, feasibility::Bool, 
                              resid_tolerance::Real, max_iter_outer::Int, max_time_outer::Real)
        return new(parse_outer_method(outer_method), parse_inner_solver(inner_solver), feasibility, 
                   resid_tolerance, max_iter_outer, max_time_outer, 
                   ConvCrit.GradientNorm, nothing, nothing, nothing, nothing, nothing)
    end
end

# Utility function to convert symbol or string to InnerSolverEnum
function parse_inner_solver(solver::Union{Symbol, String})
    solver_str = String(solver)
    if haskey(InnerSolverEnum, solver_str)
        return InnerSolverEnum[solver_str]
    else
        throw(ArgumentError("Invalid inner solver: $solver_str. Valid solvers are: LBFGS, BFGS, Newton, GradientDescent, NelderMead."))
    end
end

# Utility function to convert symbol or string to OuterMethodEnum
function parse_outer_method(method::Union{Symbol, String})
    method_str = String(method)
    if haskey(OuterMethodEnum, method_str)
        return OuterMethodEnum[method_str]
    else
        throw(ArgumentError("Invalid outer method: $method_str. Valid methods are: SEQUOIA, QPM, AugLag, IntPt."))
    end
end

# Utility function to convert symbol or string to ConvCrit
function parse_convergence_criterion(conv_crit::Union{Symbol, String})
    conv_crit_str = String(conv_crit)
    if haskey(ConvCrit, conv_crit_str)
        return ConvCrit[conv_crit_str]
    else
        throw(ArgumentError("Invalid convergence criterion: $conv_crit_str. Valid criteria are: GradientNorm, MaxIterations, MaxTime, ConstraintResidual, NormMaxIt, MaxItMaxTime, NormMaxTime, CombinedCrit, AdaptiveIterations."))
    end
end
