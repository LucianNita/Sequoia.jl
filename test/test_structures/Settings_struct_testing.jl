# Test for basic constructor functionality
@testset "SEQUOIA_Settings Constructor Tests" begin

    # Test creating a default SEQUOIA_Settings object
    @test begin
        settings = SEQUOIA_Settings()
        settings.inner_solver isa LBFGS
        settings.max_iter == 3000
        settings.max_time == Inf
        settings.resid_tolerance == 1e-6
        settings.cost_tolerance == 1e-2
        settings.cost_min == -1e10
        settings.outer_method isa SEQUOIA
        settings.feasibility == true
        settings.step_size == nothing
    end

    # Test creating with custom arguments
    @test begin
        settings = SEQUOIA_Settings(inner_solver=BFGS(), max_iter=2000, max_time=120.0,
                                    resid_tolerance=1e-8, cost_tolerance=1e-4, 
                                    cost_min=-1e5, outer_method=QPM(), feasibility=false, step_size=0.05)
        settings.inner_solver isa BFGS
        settings.max_iter == 2000
        settings.max_time == 120.0
        settings.resid_tolerance == 1e-8
        settings.cost_tolerance == 1e-4
        settings.cost_min == -1e5
        settings.outer_method isa QPM
        settings.feasibility == false
        settings.step_size == 0.05
    end

    # Test default constructor with inner_solver and outer_method
    @test begin
        settings = SEQUOIA_Settings(inner_solver=GradientDescent(), outer_method=AugLag())
        settings.inner_solver isa GradientDescent
        settings.outer_method isa AugLag
        settings.max_iter == 1000   # Default for this constructor
        settings.max_time == Inf    # Default for this constructor
    end

    @test begin
        settings = SEQUOIA_Settings(max_iter=Int64(1e9), resid_tolerance=1e-12)
        settings.max_iter == Int64(1e9)
        settings.resid_tolerance == 1e-12
    end

    @test begin
        settings = SEQUOIA_Settings(feasibility=true)
        settings.feasibility == true
        settings.outer_method isa SEQUOIA
    end
    
end

# Test for validation logic
@testset "SEQUOIA_Settings Validation Tests" begin
    # Invalid max_iter (negative)
    @test_throws ArgumentError SEQUOIA_Settings(max_iter=-5)

    # Invalid max_time (negative)
    @test_throws ArgumentError SEQUOIA_Settings(max_time=-10.0)

    # Invalid residual tolerance (zero or negative)
    @test_throws ArgumentError SEQUOIA_Settings(resid_tolerance=0)
    @test_throws ArgumentError SEQUOIA_Settings(resid_tolerance=-1e-6)

    # Invalid cost tolerance (zero or negative)
    @test_throws ArgumentError SEQUOIA_Settings(cost_tolerance=0)
    @test_throws ArgumentError SEQUOIA_Settings(cost_tolerance=-1e-3)

    # Invalid cost_min (unreasonably low)
    @test_throws ArgumentError SEQUOIA_Settings(cost_min=-1e20)

    # Invalid step_size (negative)
    @test_throws ArgumentError SEQUOIA_Settings(step_size=-0.1)

    # Invalid inner_solver (using a type that isn't defined)
    @test_throws ArgumentError SEQUOIA_Settings(inner_solver=:invalidsolver)
    
    # Invalid outer_method (using a type that isn't defined)
    @test_throws ArgumentError SEQUOIA_Settings(outer_method=:invalidmethod)
end
