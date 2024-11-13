using CUTEst
using NLPModels
@testset "Solve interface testing" begin
    # Unit test for Quadratic Penalty Method (QPM)
    @testset "Quadratic Penalty Method (QPM)" begin
        # Define the SEQUOIA problem
        problem = SEQUOIA_pb(
            2;
            x0 = [1.5, 1.5],
            constraints = x -> [x[1] + x[2] - 2, x[2] - 1],
            eqcon = [1],
            ineqcon = [2],
            jacobian = x -> [1.0 1.0; 0.0 1.0],
            objective = x -> sum(x.^2),
            gradient = x -> 2 .* x,
            solver_settings = SEQUOIA_Settings(
                :QPM, 
                :LBFGS, 
                false, 
                1e-8, 
                100, 
                10.0, 
                1e-6;
                solver_params = [1.0, 10.0, 10.0, 0.0]
            )
        )

        # Solve the problem
        solve!(problem)

        # Verify solution
        solution = problem.solution_history.iterates[end].x
        objective_value = problem.objective(solution)

        @test isapprox(solution, [1.0, 1.0], atol=1e-4)
        @test isapprox(objective_value, 2.0, atol=1e-4)
        @test problem.solution_history.iterates[end].solver_status == :first_order
    end

    # Unit test for Augmented Lagrangian Method (ALM)
    @testset "Augmented Lagrangian Method (ALM)" begin
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
                solver_params = [1.0, 10.0, 10.0, 0.0, 0.0, 0.0]
            )
        )

        # Solve the problem
        solve!(problem)

        # Verify solution
        solution = problem.solution_history.iterates[end].x
        objective_value = problem.objective(solution)

        @test isapprox(solution, [0.5, 1.5], atol=1e-4)
        @test isapprox(objective_value, 2.5, atol=1e-4)
        @test problem.solution_history.iterates[end].solver_status == :first_order
    end

    # Unit test for Interior Point Method (IPM)
    @testset "Interior Point Method (IPM)" begin
        # Define the SEQUOIA problem
        problem = SEQUOIA_pb(
            2;
            x0 = [1.5, 1.5],
            constraints = x -> [x[1] + x[2] - 2, x[2] - 1],
            eqcon = [1],
            ineqcon = [2],
            jacobian = x -> [1.0 1.0; 0.0 1.0],
            objective = x -> sum(x.^2),
            gradient = x -> 2 .* x,
            solver_settings = SEQUOIA_Settings(
                :IntPt, 
                :LBFGS, 
                false, 
                1e-8, 
                100, 
                10.0, 
                1e-6;
                solver_params = [1.0, 0.1, 0.1, 0.0, 0.0, 0.0]
            )
        )

        # Solve the problem
        solve!(problem)

        # Verify solution
        solution = problem.solution_history.iterates[end].x
        objective_value = problem.objective(solution)

        @test isapprox(solution, [1.0, 1.0], atol=1e-4)
        @test isapprox(objective_value, 2.0, atol=1e-4)
        @test problem.solution_history.iterates[end].solver_status == :first_order
    end

    # Unit test for CUTEst problem converted to SEQUOIA
    @testset "CUTEst Problem Converted to SEQUOIA" begin
        # Initialize the CUTEst problem
        problem = CUTEstModel("HS21")  # Example problem with constraints
        x = problem.meta.x0

        # Convert CUTEst problem to SEQUOIA
        sequoia_problem = cutest_to_sequoia(problem)
        sequoia_problem.solver_settings = SEQUOIA_Settings(
            :SEQUOIA,
            :LBFGS,
            false,
            1e-8,
            100,
            10.0,
            1e-8;
            solver_params = [2.0, 3.0, 0.7]
        )

        # Solve the problem
        solve!(sequoia_problem)

        # Verify solution
        solution = sequoia_problem.solution_history.iterates[end].x
        objective_value = sequoia_problem.objective(solution)

        @test isapprox(solution, [2.0, 0.0], atol=1e-3)
        @test isapprox(objective_value, -99.96, atol=1e-3)
        @test sequoia_problem.solution_history.iterates[end].solver_status == :first_order

        # Finalize CUTEst environment
        finalize(problem)
    end
end
