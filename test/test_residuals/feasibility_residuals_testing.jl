using CUTEst
# Unit Tests for `r0` and `r0_gradient!`
@testset "Feasibility Residuals Testing" begin
    # Test 1: Residual computation for a CUTEst problem
    @testset "CUTEstModel r0 Residual" begin
        problem = CUTEstModel("HS21")  # Example with constraints
        x = problem.meta.x0  # Use the initial guess for x

        expected_residual = 185.0  # Replace with expected value for your problem
        computed_residual = r0(x, problem)

        @test computed_residual ≈ expected_residual atol=1e-5

        finalize(problem)  # Finalize CUTEst environment
    end

    # Test 2: Gradient computation for a CUTEst problem
    @testset "CUTEstModel r0 Gradient" begin
        problem = CUTEstModel("HS21")  # Example with constraints
        x = problem.meta.x0  # Use the initial guess for x
        g = zeros(length(x))  # Preallocate gradient vector

        expected_gradient = [-193.0, 19.0]  # Replace with expected value for your problem
        r0_gradient!(g, x, problem)

        @test g ≈ expected_gradient atol=1e-5

        finalize(problem)  # Finalize CUTEst environment
    end

    # Test 3: Residual computation for a SEQUOIA problem
    @testset "SEQUOIA_pb r0 Residual" begin
        problem = SEQUOIA_pb(
            3;
            x0 = [1.0, 2.0, 3.0],
            constraints = x -> [x[1] + x[2] - 5, x[2] - x[3] - 1],
            eqcon = [1],
            ineqcon = [2],
            jacobian = x -> [1.0 1.0 0.0; 0.0 1.0 -1.0]
        )
        x = problem.x0  # Use the initial guess for x

        expected_residual = 4.0  # Replace with expected value for your problem
        computed_residual = r0(x, problem)

        @test computed_residual ≈ expected_residual atol=1e-5
    end

    # Test 4: Gradient computation for a SEQUOIA problem
    @testset "SEQUOIA_pb r0 Gradient" begin
        problem = SEQUOIA_pb(
            3;
            x0 = [1.0, 2.0, 3.0],
            constraints = x -> [x[1] + x[2] - 5, x[2] - x[3] - 1],
            eqcon = [1],
            ineqcon = [2],
            jacobian = x -> [1.0 1.0 0.0; 0.0 1.0 -1.0]
        )
        x = problem.x0  # Use the initial guess for x
        g = zeros(length(x))  # Preallocate gradient vector

        expected_gradient = [-2.0, -2.0, 0.0]  # Replace with expected value for your problem
        r0_gradient!(g, x, problem)

        @test g ≈ expected_gradient atol=1e-5
    end
end