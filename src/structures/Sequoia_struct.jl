using CUTEst

export SEQUOIA_pb,
       set_initial_guess!,
       set_objective!,
       set_constraints!,
       set_solver_settings!,
       update_exit_code!,
       reset_solution_history!

"""
    SEQUOIA_pb

The `SEQUOIA_pb` struct represents an optimization problem to be solved using the SEQUOIA framework. It encapsulates all necessary components of the problem, including the problem size, objective function, constraints, solver settings, and solution history.

# Fields

- `nvar::Int`: The number of variables in the optimization problem (problem dimension). Must be a positive integer.
- `x0::Union{Nothing, Vector{Float64}}`: The initial guess for the variables. If set to `nothing`, it defaults to a zero vector of size `nvar`.
- `is_minimization::Bool`: Indicates whether the problem is a minimization problem (`true`) or a maximization problem (`false`). Defaults to `true` (minimization).
- `objective::Union{Nothing, Function}`: The objective function to be minimized or maximized. Defaults to `nothing`. If set, must return a scalar value.
- `gradient::Union{Nothing, Function}`: The gradient of the objective function. Defaults to `nothing`. If not provided, it will be computed using automatic differentiation.
- `constraints::Union{Nothing, Function}`: A function that returns the vector of constraints, or `nothing` if the problem has no constraints.
- `jacobian::Union{Nothing, Function}`: The Jacobian of the constraints. Defaults to `nothing`. If not provided, it will be computed using automatic differentiation.
- `eqcon::Vector{Int}`: Indices of equality constraints, assuming `c_i(x) = 0`. Defaults to an empty vector.
- `ineqcon::Vector{Int}`: Indices of inequality constraints, assuming `c_i(x) ≤ 0`. Defaults to an empty vector.
- `solver_settings::SEQUOIA_Settings`: Solver settings defining the optimization method, tolerances, and other parameters. Defaults to a `SEQUOIA_Settings` instance with preset configurations.
- `solution_history::SEQUOIA_History`: Stores the history of solution steps for the optimization process. Defaults to an empty `SEQUOIA_History`.
- `cutest_nlp::Union{Nothing, CUTEst.CUTEstModel}`: An optional field for storing a CUTEst model instance, used for interfacing with external solvers or benchmark problems. Defaults to `nothing`.

# Constructor

## Full Constructor
    SEQUOIA_pb(nvar::Int;
               x0::Union{Nothing, Vector{Float64}} = nothing,
               is_minimization::Bool = true,
               objective::Union{Nothing, Function} = nothing,
               gradient::Union{Nothing, Function} = nothing,
               constraints::Union{Nothing, Function} = nothing,
               jacobian::Union{Nothing, Function} = nothing,
               eqcon::Vector{Int} = Int[],
               ineqcon::Vector{Int} = Int[],
               solver_settings::SEQUOIA_Settings = SEQUOIA_Settings(:QPM, :LBFGS, false, 1e-6, 1000, 300, 1e-6),
               solution_history::SEQUOIA_History = SEQUOIA_History(),
               cutest_nlp::Union{Nothing, CUTEst.CUTEstModel} = nothing)

Constructor for the `SEQUOIA_pb` struct, representing an optimization problem.

# Arguments
- `nvar::Int`: The number of variables in the problem. Must be a positive integer.
- `x0::Union{Nothing, Vector{Float64}}` (optional): The initial guess for the variables. Defaults to a zero vector of size `nvar` if not provided.
- `is_minimization::Bool` (optional): Specifies whether the problem is a minimization (`true`) or maximization (`false`). Defaults to `true`.
- `objective::Union{Nothing, Function}` (optional): The objective function for the problem. Defaults to `nothing`.
- `gradient::Union{Nothing, Function}` (optional): The gradient of the objective function. Defaults to `nothing`. If not provided, automatic differentiation will be used.
- `constraints::Union{Nothing, Function}` (optional): A function returning the vector of constraints. Defaults to `nothing`.
- `jacobian::Union{Nothing, Function}` (optional): The Jacobian matrix of the constraints. Defaults to `nothing`. If not provided, automatic differentiation will be used.
- `eqcon::Vector{Int}` (optional): A vector specifying the indices of equality constraints. Defaults to an empty vector.
- `ineqcon::Vector{Int}` (optional): A vector specifying the indices of inequality constraints. Defaults to an empty vector.
- `solver_settings::SEQUOIA_Settings` (optional): The solver settings for the optimization problem. Defaults to a preset `SEQUOIA_Settings` instance.
- `solution_history::SEQUOIA_History` (optional): The solution history for the problem. Defaults to an empty `SEQUOIA_History`.
- `cutest_nlp::Union{Nothing, CUTEst.CUTEstModel}` (optional): An optional CUTEst model instance. Defaults to `nothing`.

# Returns
A `SEQUOIA_pb` instance representing the optimization problem.

# Throws
- `ArgumentError` if `nvar` is not a positive integer.
- `ArgumentError` if the length of `x0` (if provided) does not match `nvar`.
"""
mutable struct SEQUOIA_pb
    nvar::Int                                     # Number of variables
    x0::Union{Nothing, Vector{Float64}}           # Initial guess for variables

    is_minimization::Bool                         # Whether this is a minimization problem
    objective::Union{Nothing, Function}           # Objective function
    gradient::Union{Nothing, Function}            # Gradient of the objective

    constraints::Union{Nothing, Function}         # Constraint function
    jacobian::Union{Nothing, Function}            # Jacobian of the constraints
    eqcon::Vector{Int}                            # Indices of equality constraints
    ineqcon::Vector{Int}                          # Indices of inequality constraints

    solver_settings::SEQUOIA_Settings             # Solver settings
    solution_history::SEQUOIA_History             # Solution history

    cutest_nlp::Union{Nothing, CUTEst.CUTEstModel} # Optional CUTEst model

    # Constructor
    function SEQUOIA_pb(
        nvar::Int;
        x0::Union{Nothing, Vector{Float64}} = nothing,
        is_minimization::Bool = true,
        objective::Union{Nothing, Function} = nothing,
        gradient::Union{Nothing, Function} = nothing,
        constraints::Union{Nothing, Function} = nothing,
        jacobian::Union{Nothing, Function} = nothing,
        eqcon::Vector{Int} = Int[],
        ineqcon::Vector{Int} = Int[],
        solver_settings::SEQUOIA_Settings = SEQUOIA_Settings(:QPM, :LBFGS, false, 1e-6, 1000, 300.0, 1e-6),
        solution_history::SEQUOIA_History = SEQUOIA_History(),
        cutest_nlp::Union{Nothing, CUTEst.CUTEstModel} = nothing
    )
        validate_nvar(nvar)
        if x0 === nothing
            x0 = zeros(nvar)
        else
            validate_x0(x0, nvar)
        end
        return new(nvar, x0, is_minimization, objective, gradient, constraints, jacobian, eqcon, ineqcon, solver_settings, solution_history, cutest_nlp)
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
