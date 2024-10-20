export SEQUOIA_Settings

export InnerSolverEnum, OuterMethodEnum
export ConvCrit

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

The `SEQUOIA_Settings` struct stores the key settings required for solving optimization problems using the SEQUOIA method. It allows configuration of the inner and outer solvers, convergence criteria, and various numerical tolerances. 

# Fields

- `outer_method::OuterMethodEnum`: Specifies the outer optimization method. Valid options include:
    - `SEQUOIA`: Sequential quadratic programming-based method.
    - `QPM`: Quadratic programming method.
    - `AugLag`: Augmented Lagrangian method.
    - `IntPt`: Interior point method.
  
- `inner_solver::InnerSolverEnum`: Specifies the inner solver used to solve unconstrained subproblems. Valid options include:
    - `LBFGS`: Limited-memory Broyden–Fletcher–Goldfarb–Shanno algorithm.
    - `BFGS`: Broyden–Fletcher–Goldfarb–Shanno algorithm.
    - `Newton`: Newton's method.
    - `GradientDescent`: Gradient descent method.
    - `NelderMead`: Nelder-Mead simplex method.

- `feasibility::Bool`: Determines if the solver is solving a feasibility problem (i.e., only ensuring constraints are satisfied) or optimizing an objective function.

- `resid_tolerance::Real`: Residual tolerance for constraints. The solver will consider the constraints satisfied when the residual is below this value. Must be a positive number.

- `max_iter_outer::Int`: Maximum number of iterations allowed for the outer solver. Must be a positive integer.

- `max_time_outer::Real`: Maximum computational time (in seconds) for the entire optimization process. Set to `Inf` for no time limit. Must be non-negative.

- `conv_crit::ConvCrit`: Specifies the convergence criterion for the inner solver. Valid options include:
    - `GradientNorm`: Convergence based on the gradient norm.
    - `MaxIterations`: Convergence based on the maximum number of iterations.
    - `MaxTime`: Convergence based on the maximum computational time.
    - `ConstraintResidual`: Convergence based on constraint residual.
    - Combined and adaptive criteria are also available.

- `max_iter_inner::Union{Nothing, Int}`: Maximum number of iterations for the inner solver. If `nothing`, the default behavior of the inner solver is used.

- `max_time_inner::Union{Nothing, Real}`: Maximum time (in seconds) for each inner solve call. Set to `nothing` for no limit, or a positive real number to impose a time limit.

- `cost_tolerance::Union{Nothing, Real}`: Desired optimality gap. The solver will stop when the difference between the current solution and the optimal cost is below this threshold. Must be positive if specified, or `nothing` for no specific cost tolerance.

- `cost_min::Union{Nothing, Real}`: Minimum allowed cost value. This helps detect unbounded problems. Can be set to `nothing` if not needed.

- `solver_params::Union{Nothing, Vector{Float64}}`: Optional solver-related parameters, such as step sizes, penalty parameters, or Lagrange multipliers. Defaults to `nothing` if not specified.

# Constructors

## Full Constructor
Allows complete control over all fields of the `SEQUOIA_Settings` struct.

```julia
SEQUOIA_Settings(
    outer_method::OuterMethodEnum,
    inner_solver::InnerSolverEnum,
    feasibility::Bool,
    resid_tolerance::Real,
    max_iter_outer::Int,
    max_time_outer::Real,
    conv_crit::ConvCrit,
    max_iter_inner::Union{Nothing, Int},
    max_time_inner::Union{Nothing, Real},
    cost_tolerance::Union{Nothing, Real},
    cost_min::Union{Nothing, Real},
    solver_params::Union{Nothing, Vector{Float64}} = nothing
)

# Full constructor example
settings_full = SEQUOIA_Settings(
    outer_method = QPM,
    inner_solver = LBFGS,
    feasibility = false,
    resid_tolerance = 1e-8,
    max_iter_outer = 1000,
    max_time_outer = 3600.0,
    conv_crit = GradientNorm,
    max_iter_inner = nothing,
    max_time_inner = nothing,
    cost_tolerance = 1e-4,
    cost_min = -1e6,
    solver_params = [1.0, 0.5]
)

# Minimal constructor example
settings_min = SEQUOIA_Settings(
    outer_method = QPM,
    inner_solver = LBFGS,
    feasibility = false,
    resid_tolerance = 1e-6,
    max_iter_outer = 1000,
    max_time_outer = 3600.0
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

    # Full constructor with all fields
    function SEQUOIA_Settings(outer_method::OuterMethodEnum, inner_solver::InnerSolverEnum, feasibility::Bool, 
                              resid_tolerance::Real, max_iter_outer::Int, max_time_outer::Real, conv_crit::ConvCrit, 
                              max_iter_inner::Union{Nothing, Int}, max_time_inner::Union{Nothing, Real}, 
                              cost_tolerance::Union{Nothing, Real}, cost_min::Union{Nothing, Real}, 
                              solver_params::Union{Nothing, Vector{Float64}} = nothing)
        return new(outer_method, inner_solver, feasibility, resid_tolerance, max_iter_outer, max_time_outer, 
                   conv_crit, max_iter_inner, max_time_inner, cost_tolerance, cost_min, solver_params)
    end

    # Minimal constructor requiring only key fields
    function SEQUOIA_Settings(outer_method::OuterMethodEnum, inner_solver::InnerSolverEnum, feasibility::Bool, 
                              resid_tolerance::Real, max_iter_outer::Int, max_time_outer::Real)
        return new(outer_method, inner_solver, feasibility, resid_tolerance, max_iter_outer, max_time_outer, 
                   GradientNorm, nothing, nothing, nothing, nothing, nothing)
    end
end



