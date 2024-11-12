using CUTEst
using Sequoia
using Optim
using NLPModels

# Example 1: QPM solve for a SEQUOIA problem
"""
This example demonstrates how to use `qpm_solve!` for a `SEQUOIA_pb` problem.

# Expected Output:
    QPM solve completed with time: X seconds. #X≈3.67*10^-4 on my machine
    Final solution: [1.9999609382629246, 1.874931699143694e-21]
"""
function example_qpm_solve_sequoia()
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

    println("QPM solve completed with time: $time seconds.")
    println("Final solution: $x")
end

# Example 2: QPM solve for a CUTEst problem
"""
This example demonstrates how to use `qpm_solve!` for a `CUTEstModel` problem.

# Expected Output:
    QPM solve completed with time: X seconds. # X≈5.25*10^-4 on my machine
    Final solution: [1.9999609382629246, 1.874931699143694e-21]
"""
function example_qpm_solve_cutest()
    # Initialize a CUTEst problem
    problem = CUTEstModel("HS21")  # Example with constraints
    x = problem.meta.x0

    # Convert CUTEst problem to SEQUOIA
    sequioa_problem = cutest_to_sequoia(problem)
    sequioa_problem.solver_settings = SEQUOIA_Settings(:QPM, :LBFGS, false, 10^-8, 50, 10.0, 10^-6;
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
        sequioa_problem,
        inner_solver,
        options,
        time,
        x,
        previous_fval,
        iteration,
        inner_iterations
    )

    println("QPM solve completed with time: $time seconds.")
    println("Final solution: $x")

    finalize(problem)  # Finalize CUTEst environment
end

# Run examples
example_qpm_solve_sequoia()
example_qpm_solve_cutest()
