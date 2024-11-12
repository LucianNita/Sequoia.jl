using Sequoia
using Optim
using CUTEst
using NLPModels

# Example 1: Solve a SEQUOIA problem using `sequoia_solve!`
"""
This example demonstrates how to use `sequoia_solve!` for a `SEQUOIA_pb` problem.

# Expected Output:
    SEQUOIA solve completed with time: X seconds. # X≈5.5*10^-3
    Final solution: [0.9965454393954011, 1.0033281093371909, 0.003105816383675714]
"""
function example_sequoia_solve()
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

    println("SEQUOIA solve completed with time: $time seconds.")
    println("Final solution: $x")
end

# Example 2: Solve a CUTEst problem using `sequoia_solve!`
"""
This example demonstrates how to use `sequoia_solve!` for a `CUTEstModel` problem.

# Expected Output:
    SEQUOIA solve completed with time: X seconds. # X≈4.5*10^-3
    Final solution: [1.9999951469562867, 0.002526617013478525]
"""
function example_cutest_sequoia_solve()
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

    println("SEQUOIA solve completed with time: $time seconds.")
    println("Final solution: $x")

    finalize(problem)  # Finalize CUTEst environment
end

# Run Examples
example_sequoia_solve()
example_cutest_sequoia_solve()
