using Optim
using CUTEst
using NLPModels

@testset "sequoia_solve! Testing" begin
    # Unit Test: Solve a SEQUOIA problem
    @testset "SEQUOIA solve test" begin
        # Define a SEQUOIA problem
        problem = SEQUOIA_pb(
            3;
            x0 = [1.0, 2.0, 3.0],
            constraints = x -> [x[1] + x[2] - 2, x[3] - 0.5],
            eqcon = [1],
            ineqcon = [2],
            jacobian = x -> [1.0 1.0 0.0; 0.0 0.0 1.0],
            objective = x -> sum(x.^2),
            gradient = x -> 2 .* x,
            solver_settings = SEQUOIA_Settings(
                :SEQUOIA, 
                :LBFGS, 
                false, 
                1e-8, 
                100, 
                1.0, 
                1e-8; 
                solver_params = [1.0, 2.0, 0.6], 
                cost_tolerance = 1e-8, 
                store_trace = true
            )
        )

        # Solver settings and options
        inner_solver = Optim.LBFGS()
        options = Optim.Options(iterations = 100, g_tol = 1e-6, store_trace=true, extended_trace=true)

        # Initialize variables
        time = 0.0
        x = problem.x0
        tk = problem.objective(x)
        iteration = 1
        inner_iterations = 0

        # Solve the SEQUOIA problem
        time, x, tk, iteration, inner_iterations = sequoia_solve!(
            problem,
            inner_solver,
            options,
            time,
            x,
            tk,
            iteration,
            inner_iterations
        )

        @test isapprox(x, [1.0, 1.0, 0.0]; atol=1e-2)  # Test solution
        @test problem.solution_history.iterates[end].solver_status == :first_order
        @test length(problem.solution_history.iterates) == iteration
        @test isapprox(tk, 2.0, atol=1e-3)
    end

    # Unit Test: Solve a CUTEst problem
    @testset "CUTEst solve test" begin
        # Initialize a CUTEst problem
        problem = CUTEstModel("HS21")  # Example with constraints
        x = problem.meta.x0

        # Convert CUTEst problem to SEQUOIA
        sequoia_problem = cutest_to_sequoia(problem)
        sequoia_problem.solver_settings = SEQUOIA_Settings(
            :SEQUOIA, 
            :LBFGS, 
            false, 
            1e-8, 
            100, 
            1.0, 
            1e-8; 
            solver_params = [1.0, 2.0, 0.6], 
            cost_tolerance = 1e-8, 
            store_trace = true
        )

        # Solver settings and options
        inner_solver = Optim.LBFGS()
        options = Optim.Options(iterations = 100, g_tol = 1e-6, store_trace=true, extended_trace=true)

        # Initialize variables
        time = 0.0
        tk = obj(problem, x)
        iteration = 1
        inner_iterations = 0

        # Solve the SEQUOIA problem
        time, x, tk, iteration, inner_iterations = sequoia_solve!(
            sequoia_problem,
            inner_solver,
            options,
            time,
            x,
            tk,
            iteration,
            inner_iterations
        )

        @test isapprox(x, [2.0, 0.0]; atol=1e-2)  # Test solution
        @test sequoia_problem.solution_history.iterates[end].solver_status == :first_order
        @test length(sequoia_problem.solution_history.iterates) == iteration
        @test isapprox(tk, -99.96, atol=1e-3)

        finalize(problem)  # Finalize CUTEst environment
    end
end