export SEQUOIA_Settings

# List of valid inner solvers
inner_solvers = [:LBFGS, :BFGS, :Newton, :GradientDescent, :NelderMead]

# List of valid outer methods
outer_methods = [:SEQUOIA, :QPM, :AugLag, :IntPt]

# List of valid convergence criteria
convergence_criterias = [
    :GradientNorm,       # Convergence based on gradient norm
    :MaxIterations,      # Convergence based on max number of iterations
    :MaxTime,            # Convergence based on max computational time
    :ConstraintResidual, # Convergence based on constraint residual (relevant for feasibility)
    :NormMaxIt,          # Combined: gradient norm below tolerance OR iterations exceeded
    :MaxItMaxTime,       # Combined: iterations exceeded OR max time exceeded
    :NormMaxTime,        # Combined: gradient norm below tolerance OR max time exceeded
    :CombinedCrit,       # Combined: gradient norm below tolerance OR iterations OR max time
    :AdaptiveIterations  # Adaptive inner iterations based on outer iteration count
]

"""
    SEQUOIA_Settings

The `SEQUOIA_Settings` struct stores the key settings required for solving optimization problems using the SEQUOIA method. It allows configuring various solvers, convergence criteria, and numerical settings.

# Fields

- `outer_method::Symbol`: Specifies the outer optimization method. Valid options include:
    - `:SEQUOIA`: SEQUOIA method.
    - `:QPM`: Quadratic Programming Method.
    - `:AugLag`: Augmented Lagrangian Method.
    - `:IntPt`: Interior Point Method.

- `inner_solver::Symbol`: Specifies the inner solver used to solve unconstrained subproblems. Valid options include:
    - `:LBFGS`: Limited-memory BFGS.
    - `:BFGS`: BFGS solver.
    - `:Newton`: Newton’s method.
    - `:GradientDescent`: Gradient Descent method.
    - `:NelderMead`: Nelder-Mead method.

- `feasibility::Bool`: Determines if the solver is solving a feasibility problem (true) or optimizing an objective function (false).

- `resid_tolerance::Real`: Residual tolerance for constraints. The solver considers the constraints satisfied when the residual is below this value.

- `max_iter_outer::Int`: Maximum number of iterations allowed for the outer solver.

- `max_time_outer::Real`: Maximum computational time (in seconds) for the entire optimization process.

- `conv_crit::Symbol`: Convergence criterion for the inner solver. Valid options include:
    - `:GradientNorm`: Based on the gradient norm.
    - `:MaxIterations`: Based on a maximum number of iterations.
    - `:MaxTime`: Based on maximum computational time.
    - `:ConstraintResidual`: Based on constraint residuals.
    - `:NormMaxIt`: Either gradient norm below tolerance or iterations exceeded.
    - `:MaxItMaxTime`: Either number of iterations exceeded or maximum time exceeded.
    - `:NormMaxTime`: Either gradient norm below tolerance or maximum time exceeded.
    - `:CombinedCrit`: A combination of gradient norm, number of iterations, or maximum time exceeded.
    - `:AdaptiveIterations`: Adaptive inner iterations based on outer iteration count.

- `max_iter_inner::Union{Nothing, Int}`: Maximum number of iterations for the inner solver. `Nothing` indicates that the default behavior of the inner solver is used.

- `max_time_inner::Union{Nothing, Real}`: Maximum time (in seconds) for each inner solve call. `Nothing` for no limit or a positive real number to impose a time limit.

- `cost_tolerance::Union{Nothing, Real}`: Desired optimality gap. The solver stops when the difference between the current solution and the optimal cost is below this threshold.

- `cost_min::Union{Nothing, Real}`: Minimum allowed cost value to help detect unbounded problems.

- `solver_params::Union{Nothing, Vector{Float64}}`: Optional solver-related parameters, such as step sizes, penalty parameters, or Lagrange multipliers.
"""
mutable struct SEQUOIA_Settings
    outer_method::Symbol               # Outer optimization method (:SEQUOIA, :QPM, :AugLag, :IntPt).
    inner_solver::Symbol               # The inner solver used for unconstrained problems (:LBFGS, :BFGS, etc.).
    feasibility::Bool                  # Solve feasibility problem (true) or account for an objective (false)?
    
    resid_tolerance::Real              # Residual tolerance for constraints.
    max_iter_outer::Int                # Maximum number of outer iterations allowed.
    max_time_outer::Real               # Maximum total computational time (in seconds).

    conv_crit::Symbol                  # Convergence criterion for the inner solve (:GradientNorm, etc.).
    max_iter_inner::Union{Nothing, Int}# Maximum number of inner iterations allowed.
    max_time_inner::Union{Nothing, Real}# Maximum computational time for each inner solve call (in seconds).

    cost_tolerance::Union{Nothing, Real}# Desired optimality gap.
    cost_min::Union{Nothing, Real}      # Minimum cost - useful for spotting possible unbounded problems.

    solver_params::Union{Nothing, Vector{Float64}}# Optional solver-related parameters.

    """
    SEQUOIA_Settings(outer_method::Symbol, inner_solver::Symbol, feasibility::Bool, 
                     resid_tolerance::Real, max_iter_outer::Int, max_time_outer::Real, 
                     conv_crit::Symbol, max_iter_inner::Union{Nothing, Int}, 
                     max_time_inner::Union{Nothing, Real}, cost_tolerance::Union{Nothing, Real}, 
                     cost_min::Union{Nothing, Real}, solver_params::Union{Nothing, Vector{Float64}} = nothing)

    Full constructor for `SEQUOIA_Settings`, allowing the specification of all parameters.

    # Arguments
    - `outer_method::Symbol`: The outer optimization method.
    - `inner_solver::Symbol`: The inner solver used for unconstrained problems.
    - `feasibility::Bool`: Indicates whether the solver solves a feasibility problem.
    - `resid_tolerance::Real`: Residual tolerance for constraints.
    - `max_iter_outer::Int`: Maximum number of outer iterations.
    - `max_time_outer::Real`: Maximum computational time for the outer solver.
    - `conv_crit::Symbol`: Convergence criterion for the inner solver.
    - `max_iter_inner::Union{Nothing, Int}`: Maximum number of inner iterations, or `nothing` for default behavior.
    - `max_time_inner::Union{Nothing, Real}`: Maximum time per inner solve, or `nothing` for no limit.
    - `cost_tolerance::Union{Nothing, Real}`: Desired optimality gap, or `nothing`.
    - `cost_min::Union{Nothing, Real}`: Minimum cost value to detect unbounded problems, or `nothing`.
    - `solver_params::Union{Nothing, Vector{Float64}}`: Optional solver-related parameters (default: `nothing`).
    """
    function SEQUOIA_Settings(outer_method::Symbol, inner_solver::Symbol, feasibility::Bool, 
                              resid_tolerance::Real, max_iter_outer::Int, max_time_outer::Real, 
                              conv_crit::Symbol, max_iter_inner::Union{Nothing, Int}, max_time_inner::Union{Nothing, Real}, 
                              cost_tolerance::Union{Nothing, Real}, cost_min::Union{Nothing, Real}, 
                              solver_params::Union{Nothing, Vector{Float64}} = nothing)
        

        # Create a new instance
        settings = new(outer_method, inner_solver, feasibility, 
                       resid_tolerance, max_iter_outer, max_time_outer, 
                       conv_crit, max_iter_inner, max_time_inner, 
                       cost_tolerance, cost_min, solver_params)
        
        # Call the validation function to apply defaults and check values
        validate_sequoia_settings!(settings)
        
        return settings    
    end

    """
    SEQUOIA_Settings(outer_method::Symbol, inner_solver::Symbol, feasibility::Bool, 
                     resid_tolerance::Real, max_iter_outer::Int, max_time_outer::Real)

    Minimal constructor for `SEQUOIA_Settings`. This version uses default values for some of the parameters, such as using `:GradientNorm` for `conv_crit` and `nothing` for `max_iter_inner`, `max_time_inner`, etc.

    # Arguments
    - `outer_method::Symbol`: The outer optimization method.
    - `inner_solver::Symbol`: The inner solver used for unconstrained problems.
    - `feasibility::Bool`: Indicates whether the solver solves a feasibility problem.
    - `resid_tolerance::Real`: Residual tolerance for constraints.
    - `max_iter_outer::Int`: Maximum number of outer iterations.
    - `max_time_outer::Real`: Maximum computational time for the outer solver.
    """
    function SEQUOIA_Settings(outer_method::Symbol, inner_solver::Symbol, feasibility::Bool, 
                              resid_tolerance::Real, max_iter_outer::Int, max_time_outer::Real)
        # Create a new instance with default values for the unspecified fields
        settings = new(outer_method, inner_solver, feasibility, 
                       resid_tolerance, max_iter_outer, max_time_outer, 
                       :GradientNorm, nothing, nothing, nothing, nothing, nothing)

        # Call the validation function to apply defaults and check values
        validate_sequoia_settings!(settings)

        return settings
    end

end


