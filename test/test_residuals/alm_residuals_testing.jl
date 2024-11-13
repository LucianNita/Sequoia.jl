using CUTEst
using Optim
@testset "Augmented Lagrangian residual testing" begin
    # Test 1: Augmented Lagrangian Objective for SEQUOIA Problem
    @testset "SEQUOIA Augmented Lagrangian Objective" begin
        problem = SEQUOIA_pb(
            2;
            x0 = [1.0, 2.0],
            constraints = x -> [x[1] + x[2] - 3, x[2] - 1],
            eqcon = [1],
            ineqcon = [2],
            jacobian = x -> [1.0 1.0; 0.0 1.0],
            objective = x -> sum(x.^2),
            gradient = x -> 2 .* x
        )
        μ = 1.0
        λ = [0.5, 0.5]
        x = problem.x0

        obj_value = auglag_obj(x, μ, λ, problem)
        @test isapprox(obj_value, 6.0, atol=1e-6)
    end

    # Test 2: Augmented Lagrangian Gradient for SEQUOIA Problem
    @testset "SEQUOIA Augmented Lagrangian Gradient" begin
        problem = SEQUOIA_pb(
            2;
            x0 = [1.0, 2.0],
            constraints = x -> [x[1] + x[2] - 3, x[2] - 1],
            eqcon = [1],
            ineqcon = [2],
            jacobian = x -> [1.0 1.0; 0.0 1.0],
            objective = x -> sum(x.^2),
            gradient = x -> 2 .* x
        )
        μ = 1.0
        λ = [0.5, 0.5]
        x = problem.x0
        grad_storage = zeros(length(x))

        auglag_grad!(grad_storage, x, μ, λ, problem)
        expected_grad = [2.0, 5.5]
        @test isapprox(grad_storage, expected_grad, atol=1e-6)
    end

    # Test 3: Update Lagrange Multipliers for SEQUOIA Problem
    @testset "SEQUOIA Lagrange Multiplier Update" begin
        problem = SEQUOIA_pb(
            2;
            x0 = [1.0, 2.0],
            constraints = x -> [x[1] + x[2] - 3, x[2] - 1],
            eqcon = [1],
            ineqcon = [2],
            jacobian = x -> [1.0 1.0; 0.0 1.0],
            objective = x -> sum(x.^2),
            gradient = x -> 2 .* x
        )
        μ = 1.0
        λ = [0.5, 0.5]
        x = problem.x0

        update_lag_mult!(x, μ, λ, problem)
        expected_lambdas = [0.5, 1.5]
        @test isapprox(λ, expected_lambdas, atol=1e-6)
    end

    # Test 4: Augmented Lagrangian Objective for CUTEst Problem
    @testset "CUTEst Augmented Lagrangian Objective" begin
        problem = CUTEstModel("HS21")
        x = problem.meta.x0
        μ = 1.0
        λ = [0.5, 0.5, 0.5, 0.5, 0.5]

        obj_value = auglag_obj(x, μ, λ, problem)
        @test isapprox(obj_value, 97.01, atol=1e-6)

        finalize(problem)
    end

    # Test 5: Augmented Lagrangian Gradient for CUTEst Problem
    @testset "CUTEst Augmented Lagrangian Gradient" begin
        problem = CUTEstModel("HS21")
        x = problem.meta.x0
        total_eq_con = length(problem.meta.jfix) + length(problem.meta.ifix)
        total_ineq_con = length(problem.meta.jlow) + length(problem.meta.ilow) +
                        length(problem.meta.jupp) + length(problem.meta.iupp) +
                        2 * (length(problem.meta.jrng) + length(problem.meta.irng))
        λ = ones(total_eq_con + total_ineq_con)
        μ = 10.0
        grad_storage = zeros(length(x))

        auglag_grad!(grad_storage, x, μ, λ, problem)
        expected_grad = [-1941.02, 189.0]
        @test isapprox(grad_storage, expected_grad, atol=1e-6)

        finalize(problem)
    end

    # Test 6: Update Lagrange Multipliers for CUTEst Problem
    @testset "CUTEst Lagrange Multiplier Update" begin
        problem = CUTEstModel("HS21")
        x = problem.meta.x0
        μ = 1.0
        λ = zeros(length(res(x, problem)))

        update_lag_mult!(x, μ, λ, problem)
        expected_lambdas = [19.0, 3.0, 0.0, 0.0, 0.0]
        @test isapprox(λ, expected_lambdas, atol=1e-6)

        finalize(problem)
    end
end