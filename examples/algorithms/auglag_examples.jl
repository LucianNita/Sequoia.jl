using Sequoia, Optim, CUTEst, NLPModels

"""
Example demonstrating `alm_solve!` for a `SEQUOIA_pb` problem.

Expected Output:
    ALM Solve Completed
    Final Solution: [0.5000000000000001, 1.499976984866648]
    Objective Value: 2.500950951042217
"""
function example_alm_solve_sequoia()
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

    println("ALM Solve Completed")
    println("Final Solution: $x")
    println("Objective Value: $previous_fval")
end

"""
Example demonstrating `alm_solve!` for a `CUTEstModel` problem.

Expected Output:
    ALM Solve Completed
    Final Solution: [1.9998629898908966, -4.854728251886606e-22]
    Objective Value: -99.96000274020219
"""
function example_alm_solve_cutest()
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

    println("ALM Solve Completed")
    println("Final Solution: $x")
    println("Objective Value: $previous_fval")

    # Finalize CUTEst environment
    finalize(problem)
end

example_alm_solve_sequoia()
example_alm_solve_cutest()
