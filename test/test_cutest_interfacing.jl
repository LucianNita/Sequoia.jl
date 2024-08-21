using CUTEst
using NLPModels

# Define the test set for Cutest2Sequoia
@testset "Cutest2Sequoia Tests" begin
    # Test 1: Valid problem from the CUTEst database
    @testset "Valid Problem" begin
        problem = "HS25"  # Example of a well-known test problem

        # Test if the function returns a SEQUOIA problem object
        sq_pb = Cutest2Sequoia(problem)
        @test isa(sq_pb, SEQUOIA)  # Check that the return type is SEQUOIA

        # Test if the number of variables in the SEQUOIA problem matches the CUTEst problem
        pb = CUTEstModel(problem)
        @test sq_pb.nvar == pb.meta.nvar

        # Test if the initial guess `x0` is correctly set
        @test sq_pb.x0 == pb.meta.x0

        # Test if the number of equality constraints is as expected
        expected_eq_constraints = length(pb.meta.jfix) + length(pb.meta.ifix)
        @test length(sq_pb.eqcon) == expected_eq_constraints

        # Test if the number of inequality constraints is as expected
        expected_ineq_constraints = length(pb.meta.jlow) + length(pb.meta.jupp) + 
                                    2*length(pb.meta.jrng) + length(pb.meta.ilow) + 
                                    length(pb.meta.iupp) + 2*length(pb.meta.irng)
        @test length(sq_pb.ineqcon) == expected_ineq_constraints

        finalize(pb)  # Clean up
    end

    # Test 2: Invalid problem name (problem does not exist)
    @testset "Invalid Problem" begin
        invalid_problem = "InvalidProblemName"
        @test_throws ErrorException Cutest2Sequoia(invalid_problem)
    end

    # Test 3: Problem with no constraints (or minimal constraints)
    @testset "Minimal Constraints Problem" begin
        minimal_problem = "HS3"  # Example of a simple problem with minimal constraints

        # Test if the function returns a SEQUOIA problem object
        sq_pb = Cutest2Sequoia(minimal_problem)
        @test isa(sq_pb, SEQUOIA)  # Check that the return type is SEQUOIA
        pb = CUTEstModel(minimal_problem)

        # Test if the number of equality and inequality constraints are correct
        expected_eq_constraints = 0  # Assuming no fixed constraints in HS3
        expected_ineq_constraints = length(pb.meta.jlow) + length(pb.meta.jupp) +
                                    2*length(pb.meta.jrng) + length(pb.meta.ilow) +
                                    length(pb.meta.iupp) + 2*length(pb.meta.irng)

        @test length(sq_pb.eqcon) == expected_eq_constraints
        @test length(sq_pb.ineqcon) == expected_ineq_constraints

        finalize(pb)  # Clean up
    end

    # Additional tests can be added here to cover more scenarios
end