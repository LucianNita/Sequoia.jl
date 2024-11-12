using Optim
using CUTEst
using NLPModels

# Unit Tests for `feasibility_solve!`
@testset "Sequoia Feasibility Tests" begin
    # Test 1: Feasibility solve for a SEQUOIA problem
    @testset "Feasibility Solve for SEQUOIA Problem" begin
        # Define a SEQUOIA problem
        problem = SEQUOIA_pb(
            3;
            x0 = [1.0, 2.0, 3.0],
            constraints = x -> [x[1] + x[2] - 5, x[3] - 1],
            eqcon = [1],
            ineqcon = [2],
            jacobian = x -> [1.0 1.0 0.0; 0.0 0.0 1.0],
            objective = x -> sum(x.^2),
            gradient = x -> 2 .* x
        )

        # Solver settings and options
        inner_solver = Optim.LBFGS()
        options = Optim.Options(iterations = 100, g_tol = 1e-6)

        # Initialize variables
        time = 0.0
        x = copy(problem.x0)
        previous_fval = problem.objective(x)
        inner_iterations = 0

        # Solve the feasibility problem
        time, x, previous_fval, inner_iterations = feasibility_solve!(
            problem,
            inner_solver,
            options,
            time,
            x,
            previous_fval,
            inner_iterations
        )

        @test isapprox(previous_fval, problem.objective(x), atol=1e-3)
        @test isapprox(x, [2.0, 3.0, 0.4062], atol=1e-3)
        @test problem.solution_history.iterates[end].solver_status == :small_residual
        @test inner_iterations > 0
    end

    # Test 2: Feasibility solve for a CUTEst problem
    @testset "Feasibility Solve for CUTEst Problem" begin
        # Initialize a CUTEst problem
        problem = CUTEstModel("HS21")
        x = problem.meta.x0
        inner_solver = Optim.LBFGS()
        options = Optim.Options(iterations = 100, g_tol = 1e-6)

        # Wrap CUTEst problem in SEQUOIA_pb
        sequoia_problem = cutest_to_sequoia(problem)

        # Initialize variables
        time = 0.0
        previous_fval = obj(problem, x)
        inner_iterations = 0

        # Solve the feasibility problem
        time, x, previous_fval, inner_iterations = feasibility_solve!(
            sequoia_problem,
            inner_solver,
            options,
            time,
            x,
            previous_fval,
            inner_iterations
        )

        @test isapprox(previous_fval, sequoia_problem.objective(x), atol=1e-3)
        @test isapprox(x, [50.0, -6.0207], atol=1e-3)
        @test sequoia_problem.solution_history.iterates[end].solver_status == :small_residual
        @test inner_iterations > 0

        finalize(problem)  # Finalize CUTEst environment
    end
end