using CUTEst

export SEQUOIA_pb, 
       set_objective!, set_gradient!, set_constraints!, set_jacobian!, set_bounds!, 
       set_initial_guess!, set_solver_settings!, set_feasibility!, reset_solution_history!

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

The `SEQUOIA_pb` struct defines an optimization problem to be solved using the SEQUOIA_pb optimization method. It includes fields for the problem's dimension, objective function, constraints, initial guess, and solver settings.

# Fields

- `nvar::Int`: Number of variables (problem dimension).
- `objective::Function`: Cost function to be minimized or maximized. It must be a function that takes a vector of variables and returns a scalar.
- `gradient::Union{Nothing, Function}`: Gradient of the objective function (optional). Defaults to `nothing`.
- `constraints::Union{Nothing, Function}`: Function returning a vector of constraints (optional). Defaults to `nothing`.
- `jacobian::Union{Nothing, Function}`: Jacobian of the constraints (optional). Defaults to `nothing`.
- `l_bounds::Union{Nothing, Vector{Float64}}`: Lower bounds for the constraints (optional). Defaults to `nothing`.
- `u_bounds::Union{Nothing, Vector{Float64}}`: Upper bounds for the constraints (optional). Defaults to `nothing`.
- `eqcon::Vector{Int}`: Indices of equality constraints, assumes constraints are of the form `c_i(x) = 0`. Defaults to an empty vector.
- `ineqcon::Vector{Int}`: Indices of inequality constraints, assumes constraints are of the form `c_i(x) ≤ 0`. Defaults to an empty vector.
- `is_minimization::Bool`: Whether the problem is a minimization problem (`true`) or a maximization problem (`false`). Defaults to `true`.
- `is_feasibility::Bool`: Whether the problem is only looking for feasibility (`true`) or optimizing the objective function (`false`). Defaults to `false`.
- `x0::Vector{Float64}`: Initial guess for the variables. Defaults to an empty vector.
- `p::Vector{Float64}`: Parameters for various optimization methods, such as Lagrange multipliers or penalty parameters (optional). Defaults to an empty vector.
- `solver_settings::SEQUOIA_Settings`: Settings for the solver, including the inner solver method and convergence tolerances.
- `solution::SEQUOIA_Solution_step`: Stores solution data for the current step of the optimization.
- `solution_history::SEQUOIA_Iterates`: Stores the history of all solution iterations.
- `exitCode::ExitCode`: Indicates the termination status of the solver. It can be `NotCalled`, `OptimalityReached`, `Infeasibility`, `MaxIterations`, `Unbounded`, or `SolverError`. Defaults to `ExitCode.NotCalled`.
- `cutest_nlp::Union{Nothing, CUTEstModel}`: Optional field for the CUTEst model handle. Defaults to `nothing`.

# Example

```julia
using SEQUOIA_pb

# Define the cost function
objective_fn = x -> sum(x.^2)

# Define solver settings
settings = SEQUOIA_Settings(inner_solver=BFGS(), max_iter=1000, resid_tolerance=1e-6)

# Initialize the SEQUOIA_pb problem
problem = SEQUOIA_pb(nvar=2, objective=objective_fn, solver_settings=settings)

# Output: SEQUOIA_pb object with the defined problem
"""
mutable struct SEQUOIA_pb # Sequoia problem definition struct 
    nvar::Int                                                                   # Problem dimension (number of variables)

    objective::Function                                                         # Cost function to be minimized or maximized
    gradient::Union{Nothing, Function}                                          # Gradient of the objective

    constraints::Union{Nothing, Function}                                       # Function returning a vector of constraints
    jacobian::Union{Nothing, Function}                                          # Jacobian of constraints
    l_bounds::Union{Nothing, Vector{Float64}}                                   # Lower bounds for the constraints
    u_bounds::Union{Nothing, Vector{Float64}}                                   # Upper bounds for the constraints
    eqcon::Vector{Int}                                                          # Indices of equality constraints. Assumes c_i(x)=0
    ineqcon::Vector{Int}                                                        # Indices of inequality constraints. Assumes c_i(x)≤0

    is_minimization::Bool                                                       # True for minimization, false for maximization
    is_feasibility::Bool                                                        # True for feasibility problem

    x0::Vector{Float64}                                                         # Initial guess for the variables
    
    solver_settings::SEQUOIA_Settings                                           # Solver settings
    solution::SEQUOIA_Solution_step                                             # Store solution data regarding current step
    solution_history::SEQUOIA_History                                           # Store solution history 
    exitCode::Symbol                                                            # Termination status

    cutest_nlp::Union{Nothing, CUTEst.CUTEstModel}                              # CUTEst model handle - Optional 

    """
    SEQUOIA_pb(nvar::Int, objective::Function, solver_settings::SEQUOIA_Settings)

    Construct a `SEQUOIA_pb` problem with the required fields: `nvar` (number of variables), `objective` (cost function), and `solver_settings`. This version assumes no constraints or bounds.

    # Arguments
    - `nvar::Int`: The number of variables in the optimization problem.
    - `objective::Function`: The objective function to be minimized or maximized.
    - `solver_settings::SEQUOIA_Settings`: The solver settings, including algorithms and convergence criteria.

    # Example

    ```julia
    objective_fn = x -> sum(x.^2)
    settings = SEQUOIA_Settings(inner_solver=BFGS(), max_iter=1000)

    problem = SEQUOIA_pb(2, objective_fn, settings)
    """
    function SEQUOIA_pb(nvar::Int, objective::Function, solver_settings::SEQUOIA_Settings)
        # Validate inputs
        validate_nvar(nvar)
        validate_objective(objective)
        validate_solver_settings(solver_settings)
        
        # Default initial guess `x0` and validate it
        x0 = zeros(nvar)
        validate_x0(x0, nvar)
    
        return SEQUOIA_pb(nvar, objective, nothing, nothing, nothing, nothing, Int[], Int[], true, false, x0, Float64[], solver_settings, SEQUOIA_Solution_step(), SEQUOIA_Iterates(), ExitCode.NotCalled, nothing)
    end
    

    """
    SEQUOIA_pb(nvar::Int, objective::Function, solver_settings::SEQUOIA_Settings, constraints::Function)

    Construct a `SEQUOIA_pb` problem with the objective function and constraints. No bounds are assumed in this constructor.

    # Arguments
    - `nvar::Int`: The number of variables in the problem.
    - `objective::Function`: The objective function to be minimized or maximized.
    - `solver_settings::SEQUOIA_Settings`: The solver settings, including algorithms and tolerances.
    - `constraints::Function`: A function that returns a vector of constraints.

    # Example

    ```julia
    objective_fn = x -> sum(x.^2)
    constraints_fn = x -> [x[1] + x[2] - 1.0]

    settings = SEQUOIA_Settings(inner_solver=BFGS(), max_iter=1000)

    problem = SEQUOIA_pb(2, objective_fn, settings, constraints_fn)
    """
    function SEQUOIA_pb(nvar::Int, objective::Function, solver_settings::SEQUOIA_Settings, constraints::Function)
        # Validate inputs
        validate_nvar(nvar)
        validate_objective(objective)
        validate_solver_settings(solver_settings)
        validate_constraints(constraints)
    
        # Default initial guess `x0` and solver step/iterate values
        x0 = zeros(nvar)
        validate_x0(x0, nvar)
    
        return SEQUOIA_pb(nvar, objective, nothing, constraints, nothing, nothing, Int[], Int[], true, false, x0, Float64[], solver_settings, SEQUOIA_Solution_step(), SEQUOIA_Iterates(), ExitCode.NotCalled, nothing)
    end
    

    """
    SEQUOIA_pb(nvar::Int, objective::Function, solver_settings::SEQUOIA_Settings, l_bounds::Vector{Float64}, u_bounds::Vector{Float64})

    Construct a `SEQUOIA_pb` problem with lower and upper bounds for the constraints.

    # Arguments
    - `nvar::Int`: The number of variables in the problem.
    - `objective::Function`: The objective function to be minimized or maximized.
    - `solver_settings::SEQUOIA_Settings`: The solver settings, including algorithms and tolerances.
    - `l_bounds::Vector{Float64}`: A vector of lower bounds for the constraints.
    - `u_bounds::Vector{Float64}`: A vector of upper bounds for the constraints.

    # Example

    ```julia
    objective_fn = x -> sum(x.^2)
    bounds = ([-1.0, -1.0], [1.0, 1.0])

    settings = SEQUOIA_Settings(inner_solver=BFGS(), max_iter=1000)

    problem = SEQUOIA_pb(2, objective_fn, settings, bounds)
    """
    function SEQUOIA_pb(nvar::Int, objective::Function, solver_settings::SEQUOIA_Settings, bounds::Tuple{Vector{Float64}, Vector{Float64}})
        # Validate inputs
        validate_nvar(nvar)
        validate_objective(objective)
        validate_solver_settings(solver_settings)
        validate_bounds(bounds, nvar)
        
        # Default initial guess `x0` and solver step/iterate values
        x0 = zeros(nvar)
        validate_x0(x0, nvar)
    
        return SEQUOIA_pb(nvar, objective, nothing, nothing, l_bounds, u_bounds, Int[], Int[], true, false, x0, Float64[], solver_settings, SEQUOIA_Solution_step(), SEQUOIA_Iterates(), ExitCode.NotCalled, nothing)
    end

    """
    SEQUOIA_pb(nvar::Int, objective::Function)

    Construct a `SEQUOIA_pb` problem with the default solver settings.

    # Arguments
    - `nvar::Int`: The number of variables in the problem.
    - `objective::Function`: The objective function to be minimized or maximized.

    # Example

    ```julia
    objective_fn = x -> sum(x.^2)

    problem = SEQUOIA_pb(2, objective_fn)
    """
    function SEQUOIA_pb(nvar::Int, objective::Function)
        # Validate inputs
        validate_nvar(nvar)
        validate_objective(objective)
    
        default_settings = SEQUOIA_Settings(inner_solver=BFGS(), max_iter=1000, max_time=Inf, resid_tolerance=1e-6, cost_tolerance=1e-2, cost_min=-1e10, outer_method=SEQUOIA(), feasibility=false)
        
        # Default initial guess `x0` and solver step/iterate values
        x0 = zeros(nvar)
        validate_x0(x0, nvar)
    
        return SEQUOIA_pb(nvar, objective, nothing, nothing, nothing, nothing, Int[], Int[], true, false, x0, Float64[], solver_settings, SEQUOIA_Solution_step(), SEQUOIA_Iterates(), ExitCode.NotCalled, nothing)
    end
    

end

###############################################################
## Validation functions
###############################################################

# Check that nvar is a positive integer
function validate_nvar(nvar::Int)
    if nvar <= 0
        throw(ArgumentError("The number of variables `nvar` must be a positive integer. Given: $nvar"))
    end
end

# Check that the objective function is callable
function validate_objective(objective::Function)
    if !isa(objective, Function)
        throw(ArgumentError("The `objective` must be a callable function."))
    end
end

# Check that gradient, if provided, is a callable function
function validate_gradient(gradient::Union{Nothing, Function})
    if gradient !== nothing && !isa(gradient, Function)
        throw(ArgumentError("The `gradient`, if provided, must be a callable function."))
    end
end

# Check that constraints, if provided, are a callable function
function validate_constraints(constraints::Union{Nothing, Function})
    if constraints !== nothing && !isa(constraints, Function)
        throw(ArgumentError("The `constraints`, if provided, must be a callable function."))
    end
end

# Check that jacobian, if provided, is a callable function
function validate_jacobian(jacobian::Union{Nothing, Function})
    if jacobian !== nothing && !isa(jacobian, Function)
        throw(ArgumentError("The `jacobian`, if provided, must be a callable function."))
    end
end

#Validates that the lower and upper bounds are consistent. Both bounds must be vectors of the same length as `nvar`.
function validate_bounds(l_bounds::Union{Nothing, Vector{Float64}}, u_bounds::Union{Nothing, Vector{Float64}}, nvar::Int)
    if l_bounds !== nothing && u_bounds !== nothing
        if length(l_bounds) != nvar || length(u_bounds) != nvar
            throw(ArgumentError("The lower and upper bounds must have the same length as the number of variables `nvar`. Expected length: $nvar."))
        end
    end
end

# Validate initial guess `x0`: must be a vector of the same length as `nvar`
function validate_x0(x0::Vector{Float64}, nvar::Int)
    if length(x0) != nvar
        throw(ArgumentError("The initial guess `x0` must be a vector of length `nvar`. Expected length: $nvar, but got: $(length(x0))"))
    end
end

# Validate that solver settings are of type SEQUOIA_Settings
function validate_solver_settings(solver_settings::SEQUOIA_Settings)
    if !isa(solver_settings, SEQUOIA_Settings)
        throw(ArgumentError("Invalid `solver_settings` provided. It must be of type `SEQUOIA_Settings`."))
    end
end

"""
    set_objective!(pb::SEQUOIA_pb, obj::Function)

Sets the objective function of the SEQUOIA_pb problem.

# Arguments
- `pb`: The SEQUOIA_pb problem instance.
- `obj`: The new objective function to be set.
"""
function set_objective!(pb::SEQUOIA_pb, obj::Function)
    validate_objective(obj)  # Ensure the new objective function is valid
    pb.objective = obj
    # Optionally reset some related fields or history
    pb.gradient = nothing  # Reset gradient if new objective is set
    pb.solution_history = SEQUOIA_Iterates()  # Clear solution history
end

"""
    set_gradient!(pb::SEQUOIA_pb, grad::Function)

Sets the gradient of the objective function for the SEQUOIA_pb problem.

# Arguments
- `pb`: The SEQUOIA_pb problem instance.
- `grad`: The gradient function to be set.
"""
function set_gradient!(pb::SEQUOIA_pb, grad::Function)
    validate_gradient(grad)
    pb.gradient = grad
end

"""
    set_constraints!(pb::SEQUOIA_pb, constraints::Function)

Sets the constraints for the SEQUOIA_pb problem.

# Arguments
- `pb`: The SEQUOIA_pb problem instance.
- `constraints`: The new constraints function to be set.
"""
function set_constraints!(pb::SEQUOIA_pb, constraints::Function)
    validate_constraints(constraints)
    pb.constraints = constraints
    pb.jacobian = nothing  # Reset jacobian if new constraints are set
end

"""
    set_jacobian!(pb::SEQUOIA_pb, jacobian::Function)

Sets the Jacobian of the constraints for the SEQUOIA_pb problem.

# Arguments
- `pb`: The SEQUOIA_pb problem instance.
- `jacobian`: The new Jacobian function to be set.
"""
function set_jacobian!(pb::SEQUOIA_pb, jacobian::Function)
    validate_jacobian(jacobian)
    pb.jacobian = jacobian
end

"""
    set_bounds!(pb::SEQUOIA_pb, l_bounds::Vector{Float64}, u_bounds::Vector{Float64})

Sets the lower and upper bounds for the SEQUOIA_pb problem.

# Arguments
- `pb`: The SEQUOIA_pb problem instance.
- `l_bounds`: A vector of lower bounds for the constraints.
- `u_bounds`: A vector of upper bounds for the constraints.
"""
function set_bounds!(pb::SEQUOIA_pb, l_bounds::Vector{Float64}, u_bounds::Vector{Float64})
    validate_bounds(l_bounds, u_bounds, pb.nvar)  # Ensure bounds are valid
    pb.l_bounds = l_bounds
    pb.u_bounds = u_bounds
end

"""
    set_initial_guess!(pb::SEQUOIA_pb, x0::Vector{Float64})

Sets the initial guess for the SEQUOIA_pb problem.

# Arguments
- `pb`: The SEQUOIA_pb problem instance.
- `x0`: The new initial guess vector.
"""
function set_initial_guess!(pb::SEQUOIA_pb, x0::Vector{Float64})
    validate_x0(x0, pb.nvar)
    pb.x0 = x0
end

"""
    set_solver_settings!(pb::SEQUOIA_pb, settings::SEQUOIA_Settings)

Sets the solver settings for the SEQUOIA_pb problem.

# Arguments
- `pb`: The SEQUOIA_pb problem instance.
- `settings`: The new solver settings.
"""
function set_solver_settings!(pb::SEQUOIA_pb, settings::SEQUOIA_Settings)
    validate_solver_settings(settings)
    pb.solver_settings = settings
end

"""
    set_feasibility!(pb::SEQUOIA_pb, is_feasibility::Bool)

Sets whether the SEQUOIA_pb problem is a feasibility problem.

# Arguments
- `pb`: The SEQUOIA_pb problem instance.
- `is_feasibility`: Whether the problem should focus only on feasibility (`true`) or optimization (`false`).
"""
function set_feasibility!(pb::SEQUOIA_pb, is_feasibility::Bool)
    pb.is_feasibility = is_feasibility
    pb.solution_history = SEQUOIA_Iterates()  # Reset solution history
end

"""
    reset_solution_history!(pb::SEQUOIA_pb)

Resets the solution history of the SEQUOIA_pb problem.

# Arguments
- `pb`: The SEQUOIA_pb problem instance.
"""
function reset_solution_history!(pb::SEQUOIA_pb)
    pb.solution_history = SEQUOIA_Iterates()
end