using Sequoia, CUTEst, NLPModels
"""
Example 1: 

Solve a SEQUOIA problem using the Quadratic Penalty Method (QPM).

# Expected Output:
    QPM Solve Completed
    Final Solution: [0.9999900000999989, 0.9999900000999988]
    Objective Value: 1.9999600005999916
"""
function example_solve_qpm()
    # Define a SEQUOIA problem
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

    # Print results
    println("QPM Solve Completed")
    println("Final Solution: $(problem.solution_history.iterates[end].x)")
    println("Objective Value: $(problem.objective(problem.solution_history.iterates[end].x))")
end


"""
Example 2:

Solve a SEQUOIA problem using the Augmented Lagrangian Method (ALM).

# Expected Output:
    ALM Solve Completed
    Final Solution: [0.5000000000000001, 1.5000000000300224]
    Objective Value: 2.5000000000900675
"""
function example_solve_alm()
    # Define a SEQUOIA problem
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

    # Print results
    println("ALM Solve Completed")
    println("Final Solution: $(problem.solution_history.iterates[end].x)")
    println("Objective Value: $(problem.objective(problem.solution_history.iterates[end].x))")
end

"""
Example 3:

Solve a SEQUOIA problem using the Interior Point Method (IPM).

# Expected Output:
    IPM Solve Completed
    Final Solution: [1.000000000000026, 1.0000000000000056]
    Objective Value: 2.000000000000063
"""
function example_solve_ipm()
    # Define a SEQUOIA problem
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

    # Print results
    println("IPM Solve Completed")
    println("Final Solution: $(problem.solution_history.iterates[end].x)")
    println("Objective Value: $(problem.objective(problem.solution_history.iterates[end].x))")
end

"""
Solve a CUTEst problem using `solve!` after converting it to SEQUOIA.

# Expected Output:
    CUTEst Solve Completed
    Final Solution: [1.9999953276265758, -2.938010545136885e-9]
    Objective Value: -99.96000018689472
"""
function example_solve_cutest()
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
        10.0,
        1e-8;
        solver_params = [2.0, 3.0, 0.7]
    )

    # Solve the problem
    solve!(sequoia_problem)

    # Print results
    println("CUTEst Solve Completed")
    println("Final Solution: $(sequoia_problem.solution_history.iterates[end].x)")
    println("Objective Value: $(sequoia_problem.objective(sequoia_problem.solution_history.iterates[end].x))")

    # Finalize CUTEst environment
    finalize(problem)
end

example_solve_qpm()
example_solve_alm()
example_solve_ipm()
example_solve_cutest()

