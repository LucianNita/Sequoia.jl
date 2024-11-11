using Test
using CUTEst
using Sequoia
using LinearAlgebra

# Unit tests for `qpm_obj` and `qpm_grad!`

@testset "Quadratic Penalty Objective and Gradient Tests" begin
    # Test 1: Quadratic penalty objective for a SEQUOIA problem
    @testset "SEQUOIA_pb - Objective" begin
        problem = SEQUOIA_pb(
            3;
            x0 = [1.0, 2.0, 3.0],
            objective = x -> sum(x.^2),
            gradient = x -> 2 .* x,
            constraints = x -> [x[1] + x[2] - 5, x[2] - x[3] - 1],
            eqcon = [1],
            ineqcon = [2],
            jacobian = x -> [1.0 1.0 0.0; 0.0 1.0 -1.0]
        )
        x = problem.x0
        μ = 2.5
        qpm_value = qpm_obj(x, μ, problem)
        @test isapprox(qpm_value, 24.0, atol=1e-6) 
    end

    # Test 2: Quadratic penalty gradient for a SEQUOIA problem
    @testset "SEQUOIA_pb - Gradient" begin
        problem = SEQUOIA_pb(
            3;
            x0 = [1.0, 2.0, 3.0],
            objective = x -> sum(x.^2),
            gradient = x -> 2 .* x,
            constraints = x -> [x[1] + x[2] - 5, x[2] - x[3] - 1],
            eqcon = [1],
            ineqcon = [2],
            jacobian = x -> [1.0 1.0 0.0; 0.0 1.0 -1.0]
        )
        x = problem.x0
        μ = 2.5
        g = zeros(length(x))
        qpm_grad!(g, x, μ, problem)
        expected_gradient = [-3.0, -1.0, 6.0]
        @test all(isapprox.(g, expected_gradient, atol=1e-6)) 
    end

    # Test 3: Quadratic penalty objective for a CUTEst problem
    @testset "CUTEstModel - Objective" begin
        problem = CUTEstModel("HS21")  # Example with constraints
        x = problem.meta.x0
        μ = 2.0
        qpm_value = qpm_obj(x, μ, problem)
        finalize(problem)
        @test isapprox(qpm_value, 271.01, atol=1e-6) 
    end

    # Test 4: Quadratic penalty gradient for a CUTEst problem
    @testset "CUTEstModel - Gradient" begin
        problem = CUTEstModel("HS21")  # Example with constraints
        x = problem.meta.x0
        μ = 2.0
        g = zeros(length(x))
        qpm_grad!(g, x, μ, problem)
        expected_gradient = [-386.02, 36.0]
        finalize(problem)
        @test all(isapprox.(g, expected_gradient, atol=1e-6)) 
    end
end
