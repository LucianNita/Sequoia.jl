# Solver settings struct
struct SolverSettings
    solver_type::String
    max_iterations::Int
    tolerance::Float64
    max_time::Float64  # in seconds
    step_size::Union{Nothing, Float64}  # Optional field, can be nothing
end

# Solution data struct
struct SolutionData
    iterations::Union{Nothing, Int}  # Number of iterations, may be nothing initially
    objective_value::Union{Nothing, Float64}  # Final objective value
    final_point::Union{Nothing, Vector{Float64}}  # Final solution vector
    computational_time::Union{Nothing, Float64}  # Time taken to solve
    exit_status::Union{Nothing, String}  # Status like "Converged", "Failed", etc.
end


# SEQUOIA struct with SolverSettings and SolutionData
struct SEQUOIA
    # Problem definition
    objective::Function
    gradient::Union{Nothing, Function}  # Gradient of the objective
    constraints::Union{Nothing, Function}
    jacobian::Union{Nothing, Function}  # Jacobian of constraints
    bounds::Union{Nothing, Tuple{Vector{Float64}, Vector{Float64}}}
    n_variables::Int
    is_minimization::Bool

    # Solver settings
    solver_settings::SolverSettings

    # Solution data
    solution::SolutionData
end