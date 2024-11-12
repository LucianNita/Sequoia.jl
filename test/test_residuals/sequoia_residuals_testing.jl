using CUTEst
using NLPModels

@testset "Sequoia residuals and gradients" begin 
    # Unit Tests for CUTEstModel
    @testset "CUTEstModel: Penalty Function" begin
        # Initialize CUTEst problem
        problem = CUTEstModel("HS21")
        x = problem.meta.x0
        tk = 10.0
        grad_storage = zeros(length(x))

        # Test penalty function value
        penalty_value = r(x, tk, problem)
        @test isapprox(penalty_value, 185.0; atol=1e-5)

        # Test gradient computation
        r_gradient!(grad_storage, x, tk, problem)
        @test length(grad_storage) == length(x)
        @test isapprox(grad_storage[1], -193.0; atol=1e-5)
        @test isapprox(grad_storage[2], 19.0; atol=1e-5)

        finalize(problem)
    end

    # Unit Tests for SEQUOIA_pb
    @testset "SEQUOIA_pb: Penalty Function" begin
        # Create SEQUOIA problem instance
        problem = SEQUOIA_pb(
            3;
            x0 = [1.0, 2.0, 3.0],
            objective = x -> sum(x.^2),
            gradient = x -> 2 .* x,
            constraints = x -> [x[1] + x[2] - 5],
            jacobian = x -> [1.0 1.0 0.0],
            eqcon = [1],
            ineqcon = Int[]
        )
        x = problem.x0
        tk = 10.0
        grad_storage = zeros(length(x))

        # Test penalty function value
        penalty_value = r(x, tk, problem)
        @test isapprox(penalty_value, 10.0; atol=1e-5)

        # Test gradient computation
        r_gradient!(grad_storage, x, tk, problem)
        @test length(grad_storage) == length(x)
        @test isapprox(grad_storage[1], 6.0; atol=1e-5)
        @test isapprox(grad_storage[2], 14.0; atol=1e-5)
        @test isapprox(grad_storage[3], 24.0; atol=1e-5)
    end
end
