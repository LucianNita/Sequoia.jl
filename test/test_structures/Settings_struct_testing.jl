@testset "SEQUOIA_Settings Unit Tests" begin
    # Test 1: Full Constructor - Valid Inputs
    @testset "Full Constructor Tests" begin
        settings_full = SEQUOIA_Settings(
            :SEQUOIA,          # Outer method
            :LBFGS,            # Inner solver
            false,             # Feasibility
            1e-8,              # Residual tolerance
            1000,              # Max iterations for outer solver
            3600.0,            # Max time for outer solver
            1e-5,              # Gradient norm tolerance
            conv_crit = :GradientNorm, # Convergence criterion
            max_iter_inner = 500,      # Max inner iterations
            max_time_inner = 300.0,    # Max time for inner solver
            store_trace = true,        # Enable tracing
            cost_tolerance = 1e-4,     # Desired optimality gap
            cost_min = -1e6,           # Minimum cost
            step_min = 1e-6,           # Minimum step size (valid range)
            solver_params = [1.0, 0.5] # Solver-specific parameters
        )
        @test settings_full.outer_method == :SEQUOIA
        @test settings_full.inner_solver == :LBFGS
        @test settings_full.feasibility == false
        @test settings_full.resid_tolerance == 1e-8
        @test settings_full.max_iter_outer == 1000
        @test settings_full.max_time_outer == 3600.0
        @test settings_full.gtol == 1e-5
        @test settings_full.conv_crit == :GradientNorm
        @test settings_full.max_iter_inner == 500
        @test settings_full.max_time_inner == 300.0
        @test settings_full.store_trace == true
        @test settings_full.cost_tolerance == 1e-4
        @test settings_full.cost_min == -1e6
        @test settings_full.step_min == 1e-6
        @test settings_full.solver_params == [1.0, 0.5]
    end

    # Test 2: Minimal Constructor - Valid Inputs
    @testset "Minimal Constructor Tests" begin
        settings_min = SEQUOIA_Settings(
            :QPM,              # Outer method
            :Newton,           # Inner solver
            true,              # Feasibility
            1e-6,              # Residual tolerance
            500,               # Max iterations for outer solver
            1800.0,            # Max time for outer solver
            1e-6               # Gradient norm tolerance
        )
        @test settings_min.outer_method == :QPM
        @test settings_min.inner_solver == :Newton
        @test settings_min.feasibility == true
        @test settings_min.resid_tolerance == 1e-6
        @test settings_min.max_iter_outer == 500
        @test settings_min.max_time_outer == 1800.0
        @test settings_min.gtol == 1e-6
        @test settings_min.conv_crit == :GradientNorm  # Default convergence criterion
        @test settings_min.max_iter_inner === nothing  # Defaults to nothing
        @test settings_min.max_time_inner === nothing  # Defaults to nothing
        @test settings_min.store_trace == false        # Defaults to false
        @test settings_min.cost_tolerance === nothing  # Defaults to nothing
        @test settings_min.cost_min === nothing        # Defaults to nothing
        @test settings_min.step_min === nothing        # Defaults to nothing
        @test settings_min.solver_params === nothing   # Defaults to nothing
    end

    # Test 3: Error Handling - Invalid Inputs
    @testset "Error Handling Tests" begin
        # Invalid outer method
        @test_throws ArgumentError SEQUOIA_Settings(
            :InvalidMethod,     # Invalid outer method
            :LBFGS,             # Inner solver
            false,              # Feasibility
            1e-8,               # Residual tolerance
            1000,               # Max iterations
            3600.0,             # Max time
            1e-6                # Gradient norm tolerance
        )

        # Invalid inner solver
        @test_throws ArgumentError SEQUOIA_Settings(
            :QPM,               # Outer method
            :InvalidSolver,     # Invalid inner solver
            false,              # Feasibility
            1e-8,               # Residual tolerance
            1000,               # Max iterations
            3600.0,             # Max time
            1e-6                # Gradient norm tolerance
        )

        # Invalid gradient norm tolerance
        @test_throws ArgumentError SEQUOIA_Settings(
            :QPM,               # Outer method
            :LBFGS,             # Inner solver
            false,              # Feasibility
            -1e-6,              # Invalid negative residual tolerance
            1000,               # Max iterations
            3600.0,             # Max time
            1e-6                # Gradient norm tolerance
        )

        # Invalid minimum step size
        @test_throws ArgumentError SEQUOIA_Settings(
            :SEQUOIA,           # Outer method
            :LBFGS,             # Inner solver
            false,              # Feasibility
            1e-8,               # Residual tolerance
            1000,               # Max iterations
            3600.0,             # Max time
            1e-5,               # Gradient norm tolerance
            conv_crit = :GradientNorm,
            step_min = 1e-21    # Step size below valid range
        )
    end

    # Test 4: Custom Solver Parameters - Valid Inputs
    @testset "Custom Solver Parameters Tests" begin
        settings_with_params = SEQUOIA_Settings(
            :AugLag,            # Outer method
            :GradientDescent,   # Inner solver
            false,              # Feasibility
            1e-6,               # Residual tolerance
            800,                # Max iterations
            3000.0,             # Max time
            1e-5,               # Gradient norm tolerance
            conv_crit = :MaxIterations, # Convergence criterion
            max_iter_inner = 100,       # Max inner iterations
            step_min = 1e-6,            # Minimum step size
            solver_params = [0.01, 10.0] # Custom solver parameters
        )
        @test settings_with_params.outer_method == :AugLag
        @test settings_with_params.inner_solver == :GradientDescent
        @test settings_with_params.feasibility == false
        @test settings_with_params.resid_tolerance == 1e-6
        @test settings_with_params.max_iter_outer == 800
        @test settings_with_params.max_time_outer == 3000.0
        @test settings_with_params.gtol == 1e-5
        @test settings_with_params.conv_crit == :MaxIterations
        @test settings_with_params.max_iter_inner == 100
        @test settings_with_params.max_time_inner === nothing  
        @test settings_with_params.store_trace == false        
        @test settings_with_params.cost_tolerance === nothing  
        @test settings_with_params.cost_min === nothing        
        @test settings_with_params.step_min == 1e-6
        @test settings_with_params.solver_params == [0.01, 10.0]
    end
end
