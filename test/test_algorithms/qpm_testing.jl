using Test
using Sequoia
using CUTEst
using Optim
using NLPModels

@testset "qpm_solve! Tests" begin

    # Test 1: QPM solve for a SEQUOIA problem
    @testset "SEQUOIA QPM Solve" begin
        # Define a SEQUOIA problem
        problem = SEQUOIA_pb(
            3;
            x0 = [1.0, 2.0, 3.0],
            constraints = x -> [x[1] + x[2] - 5, x[3] - 0.5],
            eqcon = [1],
            ineqcon = [2],
            jacobian = x -> [1.0 1.0 0.0; 0.0 0.0 1.0],
            objective = x -> sum(x.^2),
            gradient = x -> 2 .* x,
            solver_settings = SEQUOIA_Settings(:QPM, :LBFGS, false, 10^-8, 50, 10.0, 10^-6;
                solver_params = [1.0, 2.0, 10.0, 0],  # Penalty initialization and multiplier
                cost_tolerance = 1e-6,
                store_trace = true
            )
        )

        # Solver settings and options
        inner_solver = Optim.LBFGS()
        options = Optim.Options(iterations = 100, g_tol = 1e-6, store_trace=true, extended_trace=true)

        # Initialize variables
        time = 0.0
        x = problem.x0
        previous_fval = problem.objective(x)
        iteration = 1
        inner_iterations = 0

        # Solve the QPM problem
        time, x, previous_fval, iteration, inner_iterations = qpm_solve!(
            problem,
            inner_solver,
            options,
            time,
            x,
            previous_fval,
            iteration,
            inner_iterations
        )
        @test isapprox(x, [2.5,2.5,0.0], atol=1e-4)
        @test problem.solution_history.iterates[end].solver_status == :first_order
        @test length(problem.solution_history.iterates) == iteration+1
        @test isapprox(previous_fval, 12.5, atol=1e-4)
    end

    # Test 2: QPM solve for a CUTEst problem
    @testset "CUTEst QPM Solve" begin
        # Initialize a CUTEst problem
        problem = CUTEstModel("HS21")  # Example with constraints
        x = problem.meta.x0

        # Convert CUTEst problem to SEQUOIA
        sequoia_problem = cutest_to_sequoia(problem)
        sequoia_problem.solver_settings = SEQUOIA_Settings(:QPM, :LBFGS, false, 10^-8, 50, 10.0, 10^-6;
            solver_params = [1.0, 2.0, 10.0, 0],  # Penalty initialization and multiplier
            cost_tolerance = 1e-6,
            store_trace = true
        )

        # Solver settings and options
        inner_solver = Optim.LBFGS()
        options = Optim.Options(iterations = 100, g_tol = 1e-6, store_trace=true, extended_trace=true)

        # Initialize variables
        time = 0.0
        previous_fval = obj(problem, x)
        iteration = 1
        inner_iterations = 0

        # Solve the QPM problem
        time, x, previous_fval, iteration, inner_iterations = qpm_solve!(
            sequoia_problem,
            inner_solver,
            options,
            time,
            x,
            previous_fval,
            iteration,
            inner_iterations
        )

        @test isapprox(x, [2.0,0.0], atol=1e-4)
        @test sequoia_problem.solution_history.iterates[end].solver_status == :first_order
        @test length(sequoia_problem.solution_history.iterates) == iteration+1
        @test isapprox(previous_fval, -99.96, atol=1e-4)

        finalize(problem)  # Finalize CUTEst environment
    end

end
