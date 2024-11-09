@testset "SEQUOIA_Settings Validation Tests" begin

    # Test default values are applied for cost_tolerance and cost_min when outer_method is :SEQUOIA and feasibility is false
    @testset "Default Cost Tolerance and Min Cost for SEQUOIA" begin
        settings = SEQUOIA_Settings(
            :SEQUOIA,          # Outer method
            :LBFGS,            # Inner solver
            false,             # Feasibility: solving optimization problem
            1e-6,              # Residual tolerance
            1000,              # Max iterations for outer solver
            3600.0,            # Max time for outer solver
            1e-5;              # Gradient tolerance
            conv_crit = :MaxIterations,
            max_iter_inner = 500,
            max_time_inner = 300.0,
            cost_tolerance = nothing,
            cost_min = nothing
        )
        validate_sequoia_settings!(settings)
        @test settings.cost_tolerance == 1e-4   # Default cost tolerance applied
        @test settings.cost_min == -1e6         # Default cost minimum applied
    end

    # Test that an invalid convergence criterion raises an ArgumentError
    @testset "Invalid Convergence Criterion" begin
        @test_throws ArgumentError SEQUOIA_Settings(
            :SEQUOIA,          # Outer method
            :LBFGS,            # Inner solver
            false,             # Feasibility
            1e-6,              # Residual tolerance
            1000,              # Max iterations for outer solver
            3600.0,            # Max time for outer solver
            1e-5;              # Gradient tolerance
            conv_crit = :InvalidCrit,
            max_iter_inner = 500,
            max_time_inner = 300.0,
            cost_tolerance = 1e-4,
            cost_min = -1e6
        )
    end

    # Test max_iter_inner and max_time_inner default to valid values when required for certain convergence criteria
    @testset "Default Inner Iterations and Time for Convergence Criteria" begin
        # Test max_iter_inner defaults when required by convergence criterion
        settings_iter = SEQUOIA_Settings(
            :SEQUOIA,          # Outer method
            :LBFGS,            # Inner solver
            false,             # Feasibility
            1e-6,              # Residual tolerance
            1000,              # Max iterations for outer solver
            3600.0,            # Max time for outer solver
            1e-5;              # Gradient tolerance
            conv_crit = :MaxIterations,
            max_iter_inner = nothing,
            max_time_inner = 300.0,
            cost_tolerance = 1e-4,
            cost_min = -1e6
        )
        validate_sequoia_settings!(settings_iter)
        @test settings_iter.max_iter_inner == 500  # Default inner iterations set to 500

        # Test max_time_inner defaults when required by convergence criterion
        settings_time = SEQUOIA_Settings(
            :SEQUOIA,          # Outer method
            :LBFGS,            # Inner solver
            false,             # Feasibility
            1e-6,              # Residual tolerance
            1000,              # Max iterations for outer solver
            3600.0,            # Max time for outer solver
            1e-5;              # Gradient tolerance
            conv_crit = :MaxTime,
            max_iter_inner = 500,
            max_time_inner = nothing,
            cost_tolerance = 1e-4,
            cost_min = -1e6
        )
        validate_sequoia_settings!(settings_time)
        @test settings_time.max_time_inner == 60.0  # Default inner time set to 60 seconds
    end

    # Test validation for solver parameters
    @testset "Invalid Solver Parameters" begin
        # Test that invalid solver_params (not a vector of Float64) raises a TypeError
        @test_throws TypeError SEQUOIA_Settings(
            :SEQUOIA,          # Outer method
            :LBFGS,            # Inner solver
            false,             # Feasibility
            1e-6,              # Residual tolerance
            1000,              # Max iterations for outer solver
            3600.0,            # Max time for outer solver
            1e-5;              # Gradient tolerance
            conv_crit = :GradientNorm,
            max_iter_inner = 500,
            max_time_inner = 300.0,
            cost_tolerance = 1e-4,
            cost_min = -1e6,
            solver_params = [1.0, "invalid"]  # Invalid solver parameters (non-Float64)
        )
    end

    # Test validation of time and iterations, checking for invalid inputs
    @testset "Validation of Time and Iterations" begin
        # Test validation that max_iter_outer must be positive
        @test_throws ArgumentError SEQUOIA_Settings(
            :SEQUOIA,          # Outer method
            :LBFGS,            # Inner solver
            false,             # Feasibility
            1e-6,              # Residual tolerance
            0,                 # Invalid: zero max iterations for outer solver
            3600.0,            # Max time for outer solver
            1e-5;              # Gradient tolerance
            conv_crit = :MaxIterations,
            max_iter_inner = 500,
            max_time_inner = 300.0,
            cost_tolerance = 1e-4,
            cost_min = -1e6
        )

        # Test validation that max_time_outer must be non-negative
        @test_throws ArgumentError SEQUOIA_Settings(
            :SEQUOIA,          # Outer method
            :LBFGS,            # Inner solver
            false,             # Feasibility
            1e-6,              # Residual tolerance
            1000,              # Max iterations for outer solver
            -1.0,              # Invalid: negative max time for outer solver
            1e-5;              # Gradient tolerance
            conv_crit = :MaxIterations,
            max_iter_inner = 500,
            max_time_inner = 300.0,
            cost_tolerance = 1e-4,
            cost_min = -1e6
        )
    end
end