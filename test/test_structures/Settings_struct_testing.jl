# Unit tests for SEQUOIA_Settings
@testset "SEQUOIA_Settings Constructor - Valid Inputs" begin

    # Test creating SEQUOIA_Settings with the full constructor using symbols
    settings = SEQUOIA_Settings(:QPM, :LBFGS, false, 1e-6, 1000, 3600.0, :MaxIterations, 500, 300.0, 1e-4, -1e6, [1.0, 0.5])
    @test settings.outer_method == :QPM
    @test settings.inner_solver == :LBFGS
    @test settings.feasibility == false
    @test settings.resid_tolerance == 1e-6
    @test settings.max_iter_outer == 1000
    @test settings.max_time_outer == 3600.0
    @test settings.conv_crit == :MaxIterations
    @test settings.max_iter_inner == 500
    @test settings.max_time_inner == 300.0
    @test settings.cost_tolerance == 1e-4
    @test settings.cost_min == -1e6
    @test settings.solver_params == [1.0, 0.5]

    # Test minimal constructor with symbol inputs
    settings_min = SEQUOIA_Settings(:SEQUOIA, :BFGS, true, 1e-6, 2000, 7200.0)
    @test settings_min.outer_method == :SEQUOIA
    @test settings_min.inner_solver == :BFGS
    @test settings_min.feasibility == true
    @test settings_min.resid_tolerance == 1e-6
    @test settings_min.max_iter_outer == 2000
    @test settings_min.max_time_outer == 7200.0
    @test settings_min.conv_crit == :GradientNorm  # Default convergence criterion
    @test settings_min.max_iter_inner === nothing  # Defaults to nothing
    @test settings_min.max_time_inner === nothing  # Defaults to nothing
    @test settings_min.cost_tolerance === nothing  # Defaults to nothing
    @test settings_min.cost_min === nothing        # Defaults to nothing
    @test settings_min.solver_params === nothing   # Defaults to nothing

end

@testset "SEQUOIA_Settings Constructor - Invalid Inputs" begin

    # Test invalid inner_solver
    @test_throws ArgumentError SEQUOIA_Settings(:QPM, :InvalidSolver, false, 1e-6, 1000, 3600.0, :MaxIterations, 500, 300.0, 1e-4, -1e6)

    # Test invalid outer_method
    @test_throws ArgumentError SEQUOIA_Settings(:InvalidMethod, :LBFGS, false, 1e-6, 1000, 3600.0, :MaxIterations, 500, 300.0, 1e-4, -1e6)

    # Test invalid convergence criterion
    @test_throws ArgumentError SEQUOIA_Settings(:QPM, :LBFGS, false, 1e-6, 1000, 3600.0, :InvalidCrit, 500, 300.0, 1e-4, -1e6)

    # Test validation of negative max_iter_outer
    @test_throws ArgumentError SEQUOIA_Settings(:QPM, :LBFGS, false, 1e-6, -1000, 3600.0, :MaxIterations, 500, 300.0, 1e-4, -1e6)

    # Test validation of negative max_time_outer
    @test_throws ArgumentError SEQUOIA_Settings(:QPM, :LBFGS, false, 1e-6, 1000, -3600.0, :MaxIterations, 500, 300.0, 1e-4, -1e6)

    # Test validation of negative residual tolerance
    @test_throws ArgumentError SEQUOIA_Settings(:QPM, :LBFGS, false, -1e-6, 1000, 3600.0, :MaxIterations, 500, 300.0, 1e-4, -1e6)

    # Test validation of negative cost tolerance
    @test_throws ArgumentError SEQUOIA_Settings(:QPM, :LBFGS, false, 1e-6, 1000, 3600.0, :MaxIterations, 500, 300.0, -1e-4, -1e6)

    # Test validation of extremely low cost_min
    @test_throws ArgumentError SEQUOIA_Settings(:QPM, :LBFGS, false, 1e-6, 1000, 3600.0, :MaxIterations, 500, 300.0, 1e-4, -1e20)

end

@testset "Example 1: Full Constructor Test" begin
    # Full constructor example test
    settings_full = SEQUOIA_Settings(
        :SEQUOIA,          # Outer method
        :LBFGS,            # Inner solver
        false,             # Feasibility: solving an optimization problem, not just feasibility
        1e-8,              # Residual tolerance for constraints
        1000,              # Max iterations for outer solver
        3600.0,            # Max time for outer solver in seconds
        :GradientNorm,     # Convergence criterion: based on gradient norm
        nothing,           # Max iterations for inner solver (nothing means default)
        nothing,           # Max time for inner solver (nothing means no time limit)
        1e-4,              # Cost tolerance: the solver will stop if cost difference is below this
        -1e6,              # Minimum cost to help detect unbounded problems
        [1.0, 0.5]         # Optional solver parameters (e.g., penalty parameters, step sizes)
    )

    @test settings_full.outer_method == :SEQUOIA
    @test settings_full.inner_solver == :LBFGS
    @test settings_full.feasibility == false
    @test settings_full.resid_tolerance == 1e-8
    @test settings_full.max_iter_outer == 1000
    @test settings_full.max_time_outer == 3600.0
    @test settings_full.conv_crit == :GradientNorm
    @test settings_full.max_iter_inner === nothing
    @test settings_full.max_time_inner === nothing
    @test settings_full.cost_tolerance == 1e-4
    @test settings_full.cost_min == -1e6
    @test settings_full.solver_params == [1.0, 0.5]
end

@testset "Example 2: Minimal Constructor Test" begin
    # Minimal constructor example test
    settings_min = SEQUOIA_Settings(
        :QPM,              # Outer method
        :Newton,           # Inner solver
        true,              # Feasibility: solving a feasibility problem
        1e-6,              # Residual tolerance for constraints
        500,               # Max iterations for outer solver
        1800.0             # Max time for outer solver in seconds
    )

    @test settings_min.outer_method == :QPM
    @test settings_min.inner_solver == :Newton
    @test settings_min.feasibility == true
    @test settings_min.resid_tolerance == 1e-6
    @test settings_min.max_iter_outer == 500
    @test settings_min.max_time_outer == 1800.0
    @test settings_min.conv_crit == :GradientNorm  # Default convergence criterion
    @test settings_min.max_iter_inner === nothing  # Defaults to nothing
    @test settings_min.max_time_inner === nothing  # Defaults to nothing
    @test settings_min.cost_tolerance === nothing  # Defaults to nothing
    @test settings_min.cost_min === nothing        # Defaults to nothing
    @test settings_min.solver_params === nothing   # Defaults to nothing
end

@testset "Example 3: Error Handling with Invalid Inputs" begin
    # Test invalid outer method
    @test_throws ArgumentError SEQUOIA_Settings(
        :InvalidMethod,    # Invalid outer method
        :LBFGS,            # Inner solver
        false,             # Feasibility
        1e-8,              # Residual tolerance
        1000,              # Max iterations
        3600.0             # Max time
    )
end

@testset "Example 4: Using Custom Solver Parameters" begin
    # Test using custom solver parameters
    settings_with_params = SEQUOIA_Settings(
        :AugLag,            # Outer method
        :GradientDescent,   # Inner solver
        false,              # Feasibility
        1e-6,               # Residual tolerance
        800,                # Max iterations
        3000.0,             # Max time
        :MaxIterations,     # Convergence based on max iterations
        100,                # Max inner iterations
        nothing,            # No time limit for inner solver
        1e-5,               # Cost tolerance
        nothing,            # No minimum cost
        [0.01, 10.0]        # Custom solver parameters (step size and penalty parameter)
    )

    @test settings_with_params.outer_method == :AugLag
    @test settings_with_params.inner_solver == :GradientDescent
    @test settings_with_params.feasibility == false
    @test settings_with_params.resid_tolerance == 1e-6
    @test settings_with_params.max_iter_outer == 800
    @test settings_with_params.max_time_outer == 3000.0
    @test settings_with_params.conv_crit == :MaxIterations
    @test settings_with_params.max_iter_inner == 100
    @test settings_with_params.max_time_inner === nothing
    @test settings_with_params.cost_tolerance == 1e-5
    @test settings_with_params.cost_min === nothing
    @test settings_with_params.solver_params == [0.01, 10.0]
end