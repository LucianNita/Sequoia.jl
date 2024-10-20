# Test for basic constructor functionality
@testset "SEQUOIA_Settings Constructor Tests" begin

    # Test creating a default SEQUOIA_Settings object with the minimal constructor
    @test begin
        settings = SEQUOIA_Settings(QPM, LBFGS, true, 1e-6, 1000, 3600.0)
        settings.outer_method == QPM
        settings.inner_solver == LBFGS
        settings.feasibility == true
        settings.resid_tolerance == 1e-6
        settings.max_iter_outer == 1000
        settings.max_time_outer == 3600.0
        settings.conv_crit == GradientNorm  # Default value for conv_crit
        settings.max_iter_inner === nothing # Default value for max_iter_inner
        settings.max_time_inner === nothing # Default value for max_time_inner
        settings.cost_tolerance === nothing # Default value for cost_tolerance
        settings.cost_min === nothing       # Default value for cost_min
        settings.solver_params === nothing  # Default value for solver_params
    end

    # Test creating with custom arguments in the full constructor
    @test begin
        settings = SEQUOIA_Settings(QPM, BFGS, false, 1e-8, 2000, 120.0, MaxIterations, 500, 60.0, 1e-4, -1e5, [0.1, 0.5])
        settings.outer_method == QPM
        settings.inner_solver == BFGS
        settings.feasibility == false
        settings.resid_tolerance == 1e-8
        settings.max_iter_outer == 2000
        settings.max_time_outer == 120.0
        settings.conv_crit == MaxIterations
        settings.max_iter_inner == 500
        settings.max_time_inner == 60.0
        settings.cost_tolerance == 1e-4
        settings.cost_min == -1e5
        settings.solver_params == [0.1, 0.5]
    end

    # Test minimal constructor with only required arguments
    @test begin
        settings = SEQUOIA_Settings(AugLag, GradientDescent, true, 1e-6, 1500, 600.0)
        settings.outer_method == AugLag
        settings.inner_solver == GradientDescent
        settings.feasibility == true
        settings.resid_tolerance == 1e-6
        settings.max_iter_outer == 1500
        settings.max_time_outer == 600.0
        settings.conv_crit == GradientNorm  # Default value for conv_crit
        settings.max_iter_inner === nothing # Default value for max_iter_inner
        settings.max_time_inner === nothing # Default value for max_time_inner
        settings.cost_tolerance === nothing # Default value for cost_tolerance
        settings.cost_min === nothing       # Default value for cost_min
        settings.solver_params === nothing  # Default value for solver_params
    end
end

# Test for validation logic
@testset "SEQUOIA_Settings Validation Tests" begin
    # Invalid max_iter_outer (negative)
    @test_throws ArgumentError SEQUOIA_Settings(QPM, LBFGS, true, 1e-6, -5, 3600.0)

    # Invalid max_time_outer (negative)
    @test_throws ArgumentError SEQUOIA_Settings(QPM, LBFGS, true, 1e-6, 1000, -10.0)

    # Invalid resid_tolerance (zero or negative)
    @test_throws ArgumentError SEQUOIA_Settings(QPM, LBFGS, true, 0, 1000, 3600.0)
    @test_throws ArgumentError SEQUOIA_Settings(QPM, LBFGS, true, -1e-6, 1000, 3600.0)

    # Invalid cost_tolerance (zero or negative)
    @test_throws ArgumentError SEQUOIA_Settings(QPM, LBFGS, true, 1e-6, 1000, 3600.0, MaxIterations, 500, 60.0, 0, -1e5)
    @test_throws ArgumentError SEQUOIA_Settings(QPM, LBFGS, true, 1e-6, 1000, 3600.0, MaxIterations, 500, 60.0, -1e-3, -1e5)

    # Invalid cost_min (unreasonably low)
    @test_throws ArgumentError SEQUOIA_Settings(QPM, LBFGS, true, 1e-6, 1000, 3600.0, MaxIterations, 500, 60.0, 1e-4, -1e20)

    # Invalid solver_params (non-Float64 values)
    @test_throws ArgumentError SEQUOIA_Settings(QPM, LBFGS, true, 1e-6, 1000, 3600.0, MaxIterations, 500, 60.0, 1e-4, -1e5, ["invalid"])

    # Test proper validation for default max_iter_inner and max_time_inner when convergence criterion requires them
    @test begin
        settings = SEQUOIA_Settings(QPM, LBFGS, false, 1e-6, 1000, 3600.0, MaxIterations, nothing, nothing, 1e-4, -1e6)
        validate_sequoia_settings!(settings)
        settings.max_iter_inner == 500  # Default value for max_iter_inner when required by conv_crit
    end

    @test begin
        settings = SEQUOIA_Settings(QPM, LBFGS, false, 1e-6, 1000, 3600.0, MaxTime, nothing, nothing, 1e-4, -1e6)
        validate_sequoia_settings!(settings)
        settings.max_time_inner == 60.0  # Default value for max_time_inner when required by conv_crit
    end
end
