# Unit tests for SEQUOIA_Settings

@testset "SEQUOIA_Settings Constructor Tests" begin

    # Test creating SEQUOIA_Settings with the full constructor using symbols
    @test begin
        settings = SEQUOIA_Settings(:QPM, :LBFGS, false, 1e-6, 1000, 3600.0, :MaxIterations, 500, 300.0, 1e-4, -1e6, [1.0, 0.5])
        settings.outer_method == OuterMethodEnum.QPM
        settings.inner_solver == InnerSolverEnum.LBFGS
        settings.feasibility == false
        settings.resid_tolerance == 1e-6
        settings.max_iter_outer == 1000
        settings.max_time_outer == 3600.0
        settings.conv_crit == ConvCrit.MaxIterations
        settings.max_iter_inner == 500
        settings.max_time_inner == 300.0
        settings.cost_tolerance == 1e-4
        settings.cost_min == -1e6
        settings.solver_params == [1.0, 0.5]
    end

    # Test minimal constructor with symbol inputs
    @test begin
        settings = SEQUOIA_Settings(:SEQUOIA, :BFGS, true, 1e-6, 2000, 7200.0)
        settings.outer_method == OuterMethodEnum.SEQUOIA
        settings.inner_solver == InnerSolverEnum.BFGS
        settings.feasibility == true
        settings.resid_tolerance == 1e-6
        settings.max_iter_outer == 2000
        settings.max_time_outer == 7200.0
        settings.conv_crit == ConvCrit.GradientNorm  # Default convergence criterion
        settings.max_iter_inner === nothing  # Defaults to nothing
        settings.max_time_inner === nothing  # Defaults to nothing
        settings.cost_tolerance === nothing  # Defaults to nothing
        settings.cost_min === nothing        # Defaults to nothing
        settings.solver_params === nothing   # Defaults to nothing
    end

    # Test using strings instead of symbols
    @test begin
        settings = SEQUOIA_Settings("QPM", "LBFGS", false, 1e-6, 1000, 3600.0, "MaxIterations", 500, 300.0, 1e-4, -1e6)
        settings.outer_method == OuterMethodEnum.QPM
        settings.inner_solver == InnerSolverEnum.LBFGS
        settings.feasibility == false
        settings.resid_tolerance == 1e-6
        settings.max_iter_outer == 1000
        settings.max_time_outer == 3600.0
        settings.conv_crit == ConvCrit.MaxIterations
        settings.max_iter_inner == 500
        settings.max_time_inner == 300.0
        settings.cost_tolerance == 1e-4
        settings.cost_min == -1e6
        settings.solver_params === nothing  # Defaults to nothing
    end

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