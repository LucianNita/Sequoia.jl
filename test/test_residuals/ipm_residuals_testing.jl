using CUTEst
using NLPModels
using LinearAlgebra

# Unit Tests for IPM Objective and Gradient

@testset "IPM Objective and Gradient Tests" begin
    # Test 1: IPM Objective for SEQUOIA problem
    @testset "IPM Objective for SEQUOIA_pb" begin
        # Define a SEQUOIA problem
        problem = SEQUOIA_pb(
            3;
            x0 = [1.0, 2.0, 3.0],
            constraints = x -> [x[1] + x[2] - 5, x[3] - 0.5],
            jacobian = x -> [1.0 1.0 0.0; 0.0 0.0 1.0],
            eqcon = [1],
            ineqcon = [2],
            objective = x -> sum(x.^2),
            gradient = x -> 2 .* x
        )

        μ = 0.1  # Barrier parameter
        x_a = [problem.x0; zeros(length(problem.ineqcon) + length(problem.eqcon))]

        # Compute the IPM objective
        obj_value = ipm_obj(x_a, μ, problem)
        @test isapprox(obj_value, 66.26; atol=1e-4)
    end

    # Test 2: IPM Gradient for SEQUOIA problem
    @testset "IPM Gradient for SEQUOIA_pb" begin
        # Define a SEQUOIA problem
        problem = SEQUOIA_pb(
            3;
            x0 = [1.0, 2.0, 3.0],
            constraints = x -> [x[1] + x[2] - 5, x[3] - 0.5],
            jacobian = x -> [1.0 1.0 0.0; 0.0 0.0 1.0],
            eqcon = [1],
            ineqcon = [2],
            objective = x -> sum(x.^2),
            gradient = x -> 2 .* x
        )

        μ = 0.1  # Barrier parameter
        x_a = [problem.x0; zeros(length(problem.ineqcon) + length(problem.eqcon))]
        grad_storage = zeros(length(x_a))

        # Compute the IPM gradient
        ipm_grad!(grad_storage, x_a, μ, problem)

        @test isapprox(grad_storage, [4.0, 12.0, 29.0, 12.0, 11.5]; atol=1e-3)
    end

    # Test 3: IPM Objective for CUTEst problem
    @testset "IPM Objective for CUTEstModel" begin
        # Initialize a CUTEst problem
        problem = CUTEstModel("HS21")  # Example problem with constraints
        x = problem.meta.x0

        μ = 0.1  # Barrier parameter
        nvar = problem.meta.nvar
        eq = length(problem.meta.jfix) + length(problem.meta.ifix)
        iq = length(problem.meta.jlow) + length(problem.meta.ilow) + length(problem.meta.jupp) + length(problem.meta.iupp) + 2 * (length(problem.meta.jrng) + length(problem.meta.irng))
        x_a = [x; zeros(eq + iq)]

        # Compute the IPM objective
        obj_value = ipm_obj(x_a, μ, problem)

        @test isapprox(obj_value, 374.0504; atol=1e-4)

        finalize(problem)  # Finalize CUTEst environment
    end

    # Test 4: IPM Gradient for CUTEst problem
    @testset "IPM Gradient for CUTEstModel" begin
        # Initialize a CUTEst problem
        problem = CUTEstModel("HS21")  # Example problem with constraints
        x = problem.meta.x0

        μ = 0.1  # Barrier parameter
        nvar = problem.meta.nvar
        eq = length(problem.meta.jfix) + length(problem.meta.ifix)
        iq = length(problem.meta.jlow) + length(problem.meta.ilow) + length(problem.meta.jupp) + length(problem.meta.iupp) + 2 * (length(problem.meta.jrng) + length(problem.meta.irng))
        x_a = [x; zeros(eq + iq)]
        grad_storage = zeros(length(x_a))

        # Compute the IPM gradient
        ipm_grad!(grad_storage, x_a, μ, problem)

        @test isapprox(grad_storage, [-386.0008, 30.0, -7.4, -0.56, 13.8, 10.160000000000002, 6.200000000000001]; atol=1e-3)

        finalize(problem)  # Finalize CUTEst environment
    end
end
