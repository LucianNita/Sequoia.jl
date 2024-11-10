using NLPModels
using CUTEst

@testset "cutest_to_sequoia Tests" begin

    # Test 1: Problem with no constraints
    @testset "No Constraints" begin
        cutest_problem = CUTEstModel("ROSENBR")  # Unconstrained problem
        pb = cutest_to_sequoia(cutest_problem)
        @test pb.nvar == cutest_problem.meta.nvar
        @test pb.is_minimization == cutest_problem.meta.minimize
        @test pb.constraints === nothing
        @test pb.jacobian === nothing
        @test isempty(pb.eqcon)
        @test isempty(pb.ineqcon)
        finalize(cutest_problem)
    end

    # Test 2: Problem with only equality constraints
    @testset "Equality Constraints Only" begin
        cutest_problem = CUTEstModel("HS28")  # Only equality constraints
        pb = cutest_to_sequoia(cutest_problem)
        @test pb.nvar == cutest_problem.meta.nvar
        @test pb.is_minimization == cutest_problem.meta.minimize
        @test pb.constraints !== nothing
        @test pb.jacobian !== nothing
        @test !isempty(pb.eqcon)
        @test isempty(pb.ineqcon)
        finalize(cutest_problem)
    end

    # Test 3: Problem with only inequality constraints
    @testset "Inequality Constraints Only" begin
        cutest_problem = CUTEstModel("HS76")  # Only inequality constraints
        pb = cutest_to_sequoia(cutest_problem)
        @test pb.nvar == cutest_problem.meta.nvar
        @test pb.is_minimization == cutest_problem.meta.minimize
        @test pb.constraints !== nothing
        @test pb.jacobian !== nothing
        @test isempty(pb.eqcon)
        @test !isempty(pb.ineqcon)
        finalize(cutest_problem)
    end

    # Test 4: Problem with both equality and inequality constraints
    @testset "Equality and Inequality Constraints" begin
        cutest_problem = CUTEstModel("HS75")  # Both equality and inequality constraints
        pb = cutest_to_sequoia(cutest_problem)
        @test pb.nvar == cutest_problem.meta.nvar
        @test pb.is_minimization == cutest_problem.meta.minimize
        @test pb.constraints !== nothing
        @test pb.jacobian !== nothing
        @test !isempty(pb.eqcon)
        @test !isempty(pb.ineqcon)
        finalize(cutest_problem)
    end

    # Test 5: Large-scale problem with constraints
    @testset "Large-Scale Problem with Constraints" begin
        cutest_problem = CUTEstModel("STNQP1")  # Large-scale problem
        pb = cutest_to_sequoia(cutest_problem)
        @test pb.nvar == cutest_problem.meta.nvar
        @test pb.is_minimization == cutest_problem.meta.minimize
        @test pb.constraints !== nothing
        @test pb.jacobian !== nothing
        @test length(pb.eqcon) + length(pb.ineqcon) > 0
        finalize(cutest_problem)
    end

    # Test 6: Validate Converted `SEQUOIA_pb` Instance
    @testset "Validate Converted SEQUOIA_pb Instance" begin
        # Load the problem and convert it
        cutest_problem = CUTEstModel("HS35")  # Example problem with constraints
        pb = cutest_to_sequoia(cutest_problem)
        
        @test pb.nvar == cutest_problem.meta.nvar
        @test pb.is_minimization == cutest_problem.meta.minimize
        @test pb.constraints !== nothing
        @test pb.jacobian !== nothing
        @test length(pb.eqcon) + length(pb.ineqcon) > 0

        @test validate_pb!(pb) === nothing
        finalize(cutest_problem)

        @test_throws ErrorException cutest_problem = CUTEstModel("INVALID")
    end

end
