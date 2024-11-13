export SEQUOIA_Settings

# List of valid inner solvers
const inner_solvers = [:LBFGS, :BFGS, :Newton, :GradientDescent, :NelderMead]

# List of valid outer methods
const outer_methods = [:QPM, :AugLag, :IntPt, :SEQUOIA]

# List of valid convergence criteria
const convergence_criteria = [
    :GradientNorm,       # Convergence based on gradient norm

    :MaxIterations,      # Convergence based on max number of iterations
    :MaxTime,            # Convergence based on max computational time
    :ConstraintResidual, # Convergence based on constraint residual (for feasibility)

    :CombinedCrit,       # Combined: gradient norm OR iterations OR max time
    :AdaptiveIterations  # Adaptive inner iterations based on outer iteration count
]

"""
    SEQUOIA_Settings

The `SEQUOIA_Settings` struct stores the configuration parameters required for solving optimization 
problems using the SEQUOIA method. It allows for extensive customization of solvers, convergence 
criteria, and numerical tolerances.

# Fields

- `outer_method::Symbol`: Specifies the outer optimization method. Supported methods:
    - `:SEQUOIA`: SEQUOIA method.
    - `:QPM`: Quadratic Programming Method.
    - `:AugLag`: Augmented Lagrangian Method.
    - `:IntPt`: Interior Point Method.

- `inner_solver::Symbol`: Specifies the inner solver for unconstrained subproblems. Supported solvers:
    - `:LBFGS`: Limited-memory BFGS.
    - `:BFGS`: BFGS solver.
    - `:Newton`: Newton’s method.
    - `:GradientDescent`: Gradient Descent.
    - `:NelderMead`: Nelder-Mead method.

- `feasibility::Bool`: Indicates whether the solver is solving a feasibility problem (`true`) 
  or optimizing an objective function (`false`).

- `resid_tolerance::Float64`: Residual tolerance for constraints. The solver considers the constraints 
  satisfied when the residual falls below this value.

- `max_iter_outer::Int`: Maximum number of iterations allowed for the outer solver.

- `max_time_outer::Float64`: Maximum computational time (in seconds) for the outer solver.

- `gtol::Float64`: Gradient norm tolerance for convergence.

- `conv_crit::Symbol`: Convergence criterion for the inner solver. Supported criteria:
    - `:GradientNorm`: Convergence based on gradient norm.
    - `:MaxIterations`: Convergence based on maximum number of iterations.
    - `:MaxTime`: Convergence based on maximum computational time.
    - `:ConstraintResidual`: Convergence based on constraint residual.
    - `:CombinedCrit`: Combination of gradient norm, iteration count, or time limit.
    - `:AdaptiveIterations`: Adaptive inner iterations based on outer iteration count.

- `max_iter_inner::Union{Nothing, Int}`: Maximum number of iterations for the inner solver. 
  `nothing` indicates no explicit limit.

- `max_time_inner::Union{Nothing, Float64}`: Maximum time (in seconds) for each inner solve call. 
  `nothing` indicates no time limit.

- `store_trace::Bool`: Indicates whether to store partial iteration data for post-analysis.

- `cost_tolerance::Union{Nothing, Float64}`: Desired optimality gap. The solver stops when the 
  difference between the current solution and the optimal cost is below this threshold. 
  `nothing` indicates no explicit tolerance.

- `cost_min::Union{Nothing, Float64}`: Minimum cost value, useful for identifying unbounded problems. 
  `nothing` indicates no explicit minimum.

- `step_min::Union{Nothing, Float64}`: Minimum step size during optimization. 
  `nothing` indicates no explicit minimum.

- `solver_params::Union{Nothing, Vector{Float64}}`: Optional solver-specific parameters, such as 
  step sizes, penalty parameters, or warm-start multipliers. Default is `nothing`.

# Constructor

## Full Constructor
    SEQUOIA_Settings(outer_method::Symbol, inner_solver::Symbol, feasibility::Bool, 
                     resid_tolerance::Float64, max_iter_outer::Int, max_time_outer::Float64, 
                     gtol::Float64; 
                     conv_crit::Symbol = :GradientNorm, 
                     max_iter_inner::Union{Nothing, Int} = nothing, 
                     max_time_inner::Union{Nothing, Float64} = nothing, 
                     store_trace::Bool = false, 
                     cost_tolerance::Union{Nothing, Float64} = nothing, 
                     cost_min::Union{Nothing, Float64} = nothing, 
                     step_min::Union{Nothing, Float64} = nothing, 
                     solver_params::Union{Nothing, Vector{Float64}} = nothing)

Constructs a `SEQUOIA_Settings` instance with user-defined or default parameters. Ensures validation 
of settings.

# Arguments
- `outer_method`: Outer optimization method (e.g., `:SEQUOIA`, `:QPM`).
- `inner_solver`: Inner solver for unconstrained problems (e.g., `:LBFGS`, `:BFGS`).
- `feasibility`: Whether the solver is solving a feasibility problem.
- `resid_tolerance`: Residual tolerance for constraints.
- `max_iter_outer`: Maximum number of iterations for the outer solver.
- `max_time_outer`: Maximum time for the outer solver (in seconds).
- `gtol`: Gradient norm tolerance.
- `conv_crit`: Convergence criterion (default: `:GradientNorm`).
- `max_iter_inner`: Maximum number of iterations for the inner solver (default: `nothing`).
- `max_time_inner`: Maximum time for each inner solve call (default: `nothing`).
- `store_trace`: Whether to store iteration data (default: `false`).
- `cost_tolerance`: Desired optimality gap (default: `nothing`).
- `cost_min`: Minimum allowed cost (default: `nothing`).
- `step_min`: Minimum step size (default: `nothing`).
- `solver_params`: Optional solver-specific parameters (default: `nothing`).

# Throws
- `ArgumentError` if invalid parameters are provided.
"""
mutable struct SEQUOIA_Settings
    outer_method::Symbol                           # Outer optimization method (:SEQUOIA, :QPM, :AugLag, :IntPt).
    inner_solver::Symbol                           # The inner solver used (:LBFGS, :BFGS, etc.).
    feasibility::Bool                              # Solve feasibility problem (true) or optimize (false).

    resid_tolerance::Float64                       # Residual tolerance for constraints.
    max_iter_outer::Int                            # Maximum outer iterations allowed.
    max_time_outer::Float64                        # Maximum computational time (seconds).
    gtol::Float64                                  # Gradient norm tolerance.

    conv_crit::Symbol                              # Convergence criterion (:GradientNorm, etc.).
    max_iter_inner::Union{Nothing, Int}            # Maximum number of inner iterations allowed.
    max_time_inner::Union{Nothing, Real}           # Maximum computational time for each inner solve call (in seconds).
    store_trace::Bool                              # Store partial iteration data or not.

    cost_tolerance::Union{Nothing, Float64}        # Desired optimality gap.
    cost_min::Union{Nothing, Float64}              # Minimum cost - useful for spotting possible unbounded problems.
    step_min::Union{Nothing, Float64}              # Minimum step size 

    solver_params::Union{Nothing, Vector{Float64}} # Optional solver-related parameters. Used for warm starting. Can be lagrange multipliers, penalty parameter, objective upper bound etc.

    """
    Constructor for `SEQUOIA_Settings`.
    """
    function SEQUOIA_Settings(outer_method::Symbol, inner_solver::Symbol, feasibility::Bool, 
                              resid_tolerance::Float64, max_iter_outer::Int, max_time_outer::Float64, 
                              gtol::Float64;
                              conv_crit::Symbol = :GradientNorm,
                              max_iter_inner::Union{Nothing, Int} = nothing,
                              max_time_inner::Union{Nothing, Float64} = nothing,
                              store_trace::Bool = false,
                              cost_tolerance::Union{Nothing, Float64} = nothing,
                              cost_min::Union{Nothing, Float64} = nothing,
                              step_min::Union{Nothing, Float64} = nothing,
                              solver_params::Union{Nothing, Vector{Float64}} = nothing)::SEQUOIA_Settings

        settings = new(outer_method, inner_solver, feasibility, resid_tolerance, max_iter_outer, 
        max_time_outer, gtol, conv_crit, max_iter_inner, max_time_inner, 
        store_trace, cost_tolerance, cost_min, step_min, solver_params)
        
        # Validation logic
        validate_sequoia_settings!(settings)
        return settings
    end

end


