using Optim
using CUTEst
using NLPModels

@testset "alm_solve! Unit Tests" begin

    # Test 1: Solve a SEQUOIA problem using `alm_solve!`
    @testset "SEQUOIA Problem Test" begin
        # Define the SEQUOIA problem
        problem = SEQUOIA_pb(
            2;
            x0 = [1.0, 2.0],
            constraints = x -> [x[1] + x[2] - 2, x[1] - 0.5],
            eqcon = [1],
            ineqcon = [2],
            jacobian = x -> [1.0 1.0; 1.0 0.0],
            objective = x -> sum(x.^2),
            gradient = x -> 2 .* x,
            solver_settings = SEQUOIA_Settings(
                :AugLag,
                :LBFGS,
                false,
                1e-8,
                100,
                10.0,
                1e-6;
                solver_params = [1.0, 2.0, 0.1, 0, 0.0, 0.0],
                store_trace = true
            )
        )

        # Solver settings
        inner_solver = Optim.LBFGS()
        options = Optim.Options(iterations = 100, g_tol = 1e-6, store_trace=true, extended_trace=true)

        # Initialize variables
        time = 0.0
        x = problem.x0
        previous_fval = problem.objective(x)
        iteration = 1
        inner_iterations = 0

        # Solve the problem
        time, x, previous_fval, iteration, inner_iterations = alm_solve!(
            problem,
            inner_solver,
            options,
            time,
            x,
            previous_fval,
            iteration,
            inner_iterations
        )

        # Validate the results
        @test isapprox(x, [0.5,1.5], atol=1e-3)
        @test problem.solution_history.iterates[end].solver_status == :first_order
        @test length(problem.solution_history.iterates) == iteration+1
    end

    # Test 2: Solve a CUTEst problem using `alm_solve!`
    @testset "CUTEst Problem Test" begin
        # Initialize a CUTEst problem
        problem = CUTEstModel("HS21")  # Example with constraints
        x = problem.meta.x0

        # Convert CUTEst problem to SEQUOIA
        sequoia_problem = cutest_to_sequoia(problem)
        sequoia_problem.solver_settings = SEQUOIA_Settings(
            :AugLag,
            :LBFGS,
            false,
            1e-8,
            100,
            10.0,
            1e-6;
            solver_params = [1.0, 1.5, 0.1, 0, 0.0, 0.0, 0.0, 0.0, 0.0],
            store_trace = true
        )

        # Solver settings
        inner_solver = Optim.LBFGS()
        options = Optim.Options(iterations = 100, g_tol = 1e-6, store_trace=true, extended_trace=true)

        # Initialize variables
        time = 0.0
        previous_fval = obj(problem, x)
        iteration = 1
        inner_iterations = 0

        # Solve the problem
        time, x, previous_fval, iteration, inner_iterations = alm_solve!(
            sequoia_problem,
            inner_solver,
            options,
            time,
            x,
            previous_fval,
            iteration,
            inner_iterations
        )

        # Validate the results
        @test isapprox(x, [2.0,0.0], atol=1e-3)
        @test sequoia_problem.solution_history.iterates[end].solver_status == :first_order
        @test length(sequoia_problem.solution_history.iterates) == iteration+1

        # Finalize CUTEst environment
        finalize(problem)
    end
end
