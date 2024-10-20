# Test helper functions
function mock_objective(x)
    return sum(x.^2)
end

function mock_constraints(x)
    return [x[1] + x[2] - 1.0]
end

function mock_jacobian(x)
    return [1.0, 1.0]
end

function mock_gradient(x)
    return 2 * x
end

# Create a default solver settings for SEQUOIA_pb
default_settings = SEQUOIA_Settings(inner_solver=BFGS(), max_iter=1000, resid_tolerance=1e-6)

# Test SEQUOIA_pb Constructor with basic fields
@testset "SEQUOIA_pb Constructor" begin
    # Test basic constructor
    problem = SEQUOIA_pb(2, mock_objective, default_settings)
    @test problem.nvar == 2
    @test typeof(problem.objective) == Function
    @test problem.solver_settings == default_settings
    @test problem.exitCode == ExitCode.NotCalled
    @test problem.gradient === nothing
    @test problem.constraints === nothing
    @test problem.jacobian === nothing
    @test problem.l_bounds === nothing
    @test problem.u_bounds === nothing
    @test problem.x0 == [0.0, 0.0]

    # Test constructor with constraints
    problem_constraints = SEQUOIA_pb(2, mock_objective, default_settings, mock_constraints)
    @test typeof(problem_constraints.constraints) == Function
    @test problem_constraints.constraints([0.5, 0.5]) == [0.0]
    
    # Test constructor with bounds
    l_bounds = [-1.0, -1.0]
    u_bounds = [1.0, 1.0]
    problem_bounds = SEQUOIA_pb(2, mock_objective, default_settings, l_bounds, u_bounds)
    @test problem_bounds.l_bounds == l_bounds
    @test problem_bounds.u_bounds == u_bounds
    @test problem_bounds.x0 == [0.0, 0.0]

    # Test invalid constructor with negative `nvar`
    @test_throws ArgumentError SEQUOIA_pb(-1, mock_objective, default_settings)

    # Test invalid objective function
    @test_throws ArgumentError SEQUOIA_pb(2, "not a function", default_settings)
end

# Test Setter Functions
@testset "Setter Functions" begin
    problem = SEQUOIA_pb(2, mock_objective, default_settings)

    # Test set_objective!
    new_objective = x -> sum(x)
    set_objective!(problem, new_objective)
    @test problem.objective == new_objective
    @test problem.gradient === nothing  # Ensure gradient is reset
    @test typeof(problem.solution_history) == SEQUOIA_Iterates  # Ensure history is reset

    # Test invalid objective
    @test_throws ArgumentError set_objective!(problem, "not a function")

    # Test set_gradient!
    set_gradient!(problem, mock_gradient)
    @test problem.gradient == mock_gradient

    # Test invalid gradient
    @test_throws ArgumentError set_gradient!(problem, "not a function")

    # Test set_constraints!
    set_constraints!(problem, mock_constraints)
    @test typeof(problem.constraints) == Function
    @test problem.constraints([0.5, 0.5]) == [0.0]
    @test problem.jacobian === nothing  # Ensure jacobian is reset

    # Test invalid constraints
    @test_throws ArgumentError set_constraints!(problem, "not a function")

    # Test set_jacobian!
    set_jacobian!(problem, mock_jacobian)
    @test problem.jacobian == mock_jacobian

    # Test invalid jacobian
    @test_throws ArgumentError set_jacobian!(problem, "not a function")

    # Test set_bounds!
    new_l_bounds = [-2.0, -2.0]
    new_u_bounds = [2.0, 2.0]
    set_bounds!(problem, new_l_bounds, new_u_bounds)
    @test problem.l_bounds == new_l_bounds
    @test problem.u_bounds == new_u_bounds

    # Test invalid bounds with wrong dimensions
    @test_throws ArgumentError set_bounds!(problem, [-1.0], [1.0])

    # Test set_initial_guess!
    new_x0 = [0.5, 0.5]
    set_initial_guess!(problem, new_x0)
    @test problem.x0 == new_x0

    # Test invalid initial guess with wrong size
    @test_throws ArgumentError set_initial_guess!(problem, [0.5])

    # Test set_solver_settings!
    new_settings = SEQUOIA_Settings(inner_solver=LBFGS(), max_iter=500, resid_tolerance=1e-4)
    set_solver_settings!(problem, new_settings)
    @test problem.solver_settings == new_settings

    # Test invalid solver settings
    @test_throws ArgumentError set_solver_settings!(problem, "invalid settings")

    # Test set_feasibility!
    set_feasibility!(problem, true)
    @test problem.is_feasibility == true
    @test typeof(problem.solution_history) == SEQUOIA_Iterates  # Ensure history is reset

    # Test reset_solution_history!
    reset_solution_history!(problem)
    @test typeof(problem.solution_history) == SEQUOIA_Iterates  # Solution history should be reset
end

# Edge case testing
@testset "Edge Cases" begin
    # Test with 1 variable
    problem = SEQUOIA_pb(1, mock_objective, default_settings)
    @test problem.nvar == 1
    @test problem.x0 == [0.0]

    # Test large problem with 1000 variables
    problem_large = SEQUOIA_pb(1000, mock_objective, default_settings)
    @test problem_large.nvar == 1000
    @test length(problem_large.x0) == 1000
    @test problem_large.x0 == zeros(1000)

    # Test empty constraints and jacobian
    problem_empty = SEQUOIA_pb(2, mock_objective, default_settings)
    set_constraints!(problem_empty, x -> [])
    set_jacobian!(problem_empty, x -> [])
    @test problem_empty.constraints([]) == []
    @test problem_empty.jacobian([]) == []
end
