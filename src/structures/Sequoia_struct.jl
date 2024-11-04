using CUTEst

export SEQUOIA_pb,
       set_initial_guess!,
       set_objective!,
       set_constraints!,
       set_solver_settings!,
       update_exit_code!,
       reset_solution_history!


ExitCode = [
    :NotCalled,                # Problem not yet optimized
    :OptimalityReached,        # Optimal solution found
    :Infeasibility,            # Problem is infeasible
    :MaxIterations,            # Reached maximum number of iterations
    :Unbounded,                # Problem is unbounded
    :SolverError,              # Solver encountered an error
]

"""
    SEQUOIA_pb

The `SEQUOIA_pb` struct defines an optimization problem to be solved using SEQUOIA. It includes fields for the problem dimension (`nvar`), the objective function, constraints, bounds, solver settings, and solution history.

# Fields

- `nvar::Int`: The number of variables in the optimization problem (problem dimension).
- `x0::Vector{Float64}`: Initial guess for the variables, defaulting to a zero vector of size `nvar`.
- `is_minimization::Bool`: A flag indicating whether the problem is a minimization problem (`true`) or a maximization problem (`false`). Defaults to `true` (minimization).
- `objective::Union{Nothing, Function}`: The objective function to be minimized or maximized. Defaults to `nothing`.
- `gradient::Union{Nothing, Function}`: The gradient of the objective function. Defaults to `nothing`.
- `constraints::Union{Nothing, Function}`: A function returning a vector of constraints. Defaults to `nothing`.
- `jacobian::Union{Nothing, Function}`: The Jacobian of the constraints. Defaults to `nothing`.
- `eqcon::Vector{Int}`: Indices of equality constraints, assuming `c_i(x) = 0`. Defaults to an empty vector.
- `ineqcon::Vector{Int}`: Indices of inequality constraints, assuming `c_i(x) ≤ 0`. Defaults to an empty vector.
- `solver_settings::SEQUOIA_Settings`: Settings for the optimization solver, including the algorithm to use and tolerance levels. Defaults to a pre-defined set of settings.
- `solution_history::SEQUOIA_History`: Stores the history of all solution iterations. Defaults to an empty `SEQUOIA_History` object.
- `exitCode::Symbol`: The termination status of the solver, defaulting to `:NotCalled`.
- `cutest_nlp::Union{Nothing, CUTEst.CUTEstModel}`: An optional field to store a handle to a CUTEst model, used for interfacing with external solvers or benchmarks. Defaults to `nothing`.
"""
mutable struct SEQUOIA_pb
    nvar::Int
    x0::Vector{Float64}
    
    is_minimization::Bool
    objective::Union{Nothing, Function}
    gradient::Union{Nothing, Function}

    constraints::Union{Nothing, Function}
    jacobian::Union{Nothing, Function}
    eqcon::Vector{Int}
    ineqcon::Vector{Int}

    solver_settings::SEQUOIA_Settings
    solution_history::SEQUOIA_History
    exitCode::Symbol

    cutest_nlp::Union{Nothing, CUTEst.CUTEstModel}

    """
    # Constructor

    The `SEQUOIA_pb` constructor requires only the number of variables `nvar`, and provides default values for all other fields. It allows the creation of an optimization problem with varying levels of complexity, from a simple unconstrained optimization to a more complex problem with constraints and a defined solver setup.

    # Arguments

    - `nvar::Int`: The number of variables in the problem (required).
    - `x0::Vector{Float64}`: Initial guess for the variables. Defaults to a zero vector of size `nvar`.
    - `is_minimization::Bool`: A flag for minimization (`true`) or maximization (`false`). Defaults to `true`.
    - `objective::Union{Nothing, Function}`: The objective function for the problem. Defaults to `nothing`.
    - `gradient::Union{Nothing, Function}`: The gradient of the objective function. Defaults to `nothing`.
    - `constraints::Union{Nothing, Function}`: A function returning the vector of constraints. Defaults to `nothing`.
    - `jacobian::Union{Nothing, Function}`: The Jacobian matrix of the constraints. Defaults to `nothing`.
    - `eqcon::Vector{Int}`: Indices of equality constraints. Defaults to an empty vector.
    - `ineqcon::Vector{Int}`: Indices of inequality constraints. Defaults to an empty vector.
    - `solver_settings::SEQUOIA_Settings`: The settings for the solver. Defaults to `SEQUOIA_Settings(:QPM,:LBFGS,false,10^-6,1000,300)`.
    - `solution_history::SEQUOIA_History`: The history of solution steps. Defaults to an empty `SEQUOIA_History`.
    - `exitCode::Symbol`: The termination status of the solver. Defaults to `:NotCalled`.
    - `cutest_nlp::Union{Nothing, CUTEst.CUTEstModel}`: Optional CUTEst model handle. Defaults to `nothing`.
    """
    function SEQUOIA_pb(nvar::Int; 
                        x0::Vector{Float64} = zeros(nvar),
                        is_minimization::Bool = true,
                        objective::Union{Nothing, Function} = nothing,
                        gradient::Union{Nothing, Function} = nothing,
                        constraints::Union{Nothing, Function} = nothing,
                        jacobian::Union{Nothing, Function} = nothing,
                        eqcon::Vector{Int} = Int[],
                        ineqcon::Vector{Int} = Int[],
                        solver_settings::SEQUOIA_Settings = SEQUOIA_Settings(:QPM,:LBFGS,false,10^-6,1000,300,10^-6),
                        solution_history::SEQUOIA_History = SEQUOIA_History(),
                        exitCode::Symbol = :NotCalled,
                        cutest_nlp::Union{Nothing, CUTEst.CUTEstModel} = nothing)
        validate_nvar(nvar)
        return new(nvar, x0, is_minimization, objective, gradient, constraints, jacobian,
                   eqcon, ineqcon, solver_settings, solution_history, exitCode, cutest_nlp)
    end
end

###############################################################
## Setter functions
###############################################################

"""
    set_initial_guess!(pb::SEQUOIA_pb, x0::Vector{Float64})

Sets the initial guess (`x0`) for the SEQUOIA_pb problem. This function validates the input
to ensure the length of `x0` matches the problem dimension (`nvar`).

# Arguments
- `pb`: The `SEQUOIA_pb` problem instance.
- `x0`: The new initial guess vector. Must have a length equal to `pb.nvar`.

# Throws
- `ArgumentError`: If the length of `x0` does not match `pb.nvar`.
"""
function set_initial_guess!(pb::SEQUOIA_pb, x0::Vector{Float64})
    validate_x0(x0, pb.nvar)
    pb.x0 = x0
end

"""
    set_objective!(pb::SEQUOIA_pb, obj::Function; gradient::Union{Nothing, Function}=nothing, reset_history::Bool=true)

Sets the objective function for the SEQUOIA_pb problem and optionally resets the solution history.
You can also provide a gradient, otherwise, the gradient will be generated automatically.

# Arguments
- `pb`: The `SEQUOIA_pb` problem instance.
- `obj`: The new objective function, which must return a scalar value.
- `gradient`: (Optional) The gradient function. If not provided, it will be auto-generated using automatic differentiation.
- `reset_history`: (Optional) A boolean flag indicating whether to reset the solution history. Defaults to `true`.

# Throws
- `ArgumentError`: If the objective function is not callable or its output is not a scalar.
"""
function set_objective!(pb::SEQUOIA_pb, obj::Function; gradient::Union{Nothing, Function}=nothing, reset_history::Bool = true)
    validate_objective(obj,pb.x0)
    pb.objective = obj
    
    pb.gradient = gradient
    validate_gradient!(pb)

    if reset_history
        pb.solution_history = SEQUOIA_History()
    end 
end

"""
    set_objective!(pb::Any, obj::Any; gradient::Union{Nothing, Function} = nothing, reset_history::Bool = true)

Sets the objective function for a `SEQUOIA_pb` optimization problem. This function validates the problem instance and objective function before setting them. Optionally, a gradient can be provided, and the solution history can be reset.

# Arguments
- `pb`: The optimization problem instance. Must be of type `SEQUOIA_pb`.
- `obj`: The objective function to set. Must be a callable function.
- `gradient`: (Optional) The gradient function associated with the objective. Defaults to `nothing`. If not provided, a gradient may be computed using automatic differentiation.
- `reset_history`: (Optional) A boolean flag indicating whether to reset the solution history. Defaults to `true`.

# Throws
- `ArgumentError`: If `pb` is not of type `SEQUOIA_pb`.
- `ArgumentError`: If `obj` is not a callable function.
"""

function set_objective!(pb::Any, obj::Any; gradient::Union{Nothing, Function}=nothing, reset_history::Bool = true)
    pb_fallback(pb);
    objective_setter_fallback(obj);
end

"""
    set_constraints!(pb::SEQUOIA_pb, constraints::Function, eqcon::Vector{Int}, ineqcon::Vector{Int}; jacobian::Union{Nothing, Function}=nothing, reset_history::Bool=true)

Sets the constraints and optional Jacobian for the SEQUOIA_pb problem. You can also specify equality and inequality constraint indices. Optionally resets the solution history.

# Arguments
- `pb`: The `SEQUOIA_pb` problem instance.
- `constraints`: The new constraints function, which must return a vector of constraints.
- `eqcon`: A vector of indices corresponding to equality constraints (i.e., `c_i(x) = 0`). Defaults to `nothing`.
- `ineqcon`: A vector of indices corresponding to inequality constraints (i.e., `c_i(x) ≤ 0`). Defaults to `nothing`.
- `jacobian`: (Optional) The Jacobian function. If not provided, it will be auto-generated using automatic differentiation.
- `reset_history`: (Optional) A boolean flag indicating whether to reset the solution history. Defaults to `true`.

# Throws
- `ArgumentError`: If the constraints are not callable or their dimensions do not match the provided indices.
"""
function set_constraints!(pb::SEQUOIA_pb, constraints::Function, eqcon::Vector{Int}, ineqcon::Vector{Int}; jacobian::Union{Nothing, Function}=nothing, reset_history::Bool = true)
    pb.constraints = constraints
    pb.jacobian = jacobian
    pb.eqcon = eqcon
    pb.ineqcon = ineqcon
    validate_constraints!(pb)

    if reset_history
        pb.solution_history = SEQUOIA_History()
    end
end

"""
    set_solver_settings!(pb::SEQUOIA_pb, settings::SEQUOIA_Settings)

Sets the solver settings for the SEQUOIA_pb problem.

# Arguments
- `pb`: The `SEQUOIA_pb` problem instance.
- `settings`: The new solver settings (`SEQUOIA_Settings`).
"""
function set_solver_settings!(pb::SEQUOIA_pb, settings::SEQUOIA_Settings)
    pb.solver_settings = settings
end

"""
    set_solver_settings!(pb::Any, settings::Any)

Sets the solver settings for a SEQUOIA problem. This function first validates that the problem instance `pb` is of type `SEQUOIA_pb` and that the `settings` are valid `SEQUOIA_Settings`. If any validation fails, an error is thrown.

# Arguments
- `pb`: The problem instance to which the solver settings are being applied. Must be of type `SEQUOIA_pb`.
- `settings`: The solver settings to be applied. Must be of type `SEQUOIA_Settings`.

# Throws
- `ArgumentError`: If `pb` is not of type `SEQUOIA_pb`.
- `ArgumentError`: If `settings` is not of type `SEQUOIA_Settings`.
"""
function set_solver_settings!(pb::Any, settings::Any)
    pb_fallback(pb);
    solver_settings_fallback(settings);
end

"""
    update_exit_code!(pb::SEQUOIA_pb, code::Symbol)

Updates the `exitCode` field of the SEQUOIA_pb problem. This function validates that the exit code is one of the predefined, accepted codes.

# Arguments
- `pb`: The `SEQUOIA_pb` problem instance.
- `code`: The new exit code. Must be one of `:NotCalled`, `:OptimalityReached`, `:Infeasibility`, `:MaxIterations`, `:Unbounded`, `:SolverError`.

# Throws
- `ArgumentError`: If the `code` is not a valid exit code.
"""
function update_exit_code!(pb::SEQUOIA_pb, code::Symbol)
    validate_code(code)
    pb.exitCode = code
end

"""
    reset_solution_history!(pb::SEQUOIA_pb)

Resets the solution history for the SEQUOIA_pb problem.

# Arguments
- `pb`: The `SEQUOIA_pb` problem instance.
"""
function reset_solution_history!(pb::SEQUOIA_pb)
    pb.solution_history = SEQUOIA_History()
end
