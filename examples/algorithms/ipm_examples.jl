using Sequoia
using Optim
using CUTEst
using NLPModels

# Example 1: Solve a SEQUOIA problem using `ipm_solve!`
"""
This example demonstrates how to use `ipm_solve!` for a `SEQUOIA_pb` problem.

# Expected Output:
    IPM solve completed with time: X seconds. # X≈2.05*10^-3 on my machine
    Final solution: [2.5000000088084504, 2.5000000088084504, 9.761015253389142e-5]
"""
function example_ipm_solve_sequoia()
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
        solver_settings = SEQUOIA_Settings(
            :IntPt, 
            :LBFGS, 
            false, 
            1e-8, 
            100, 
            10.0, 
            1e-6; 
            solver_params = [1.0, 0.5, 0.1, 0, 0.0, 0.0], 
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
    iteration = 0
    inner_iterations = 0

    # Solve the IPM problem
    time, x, previous_fval, iteration, inner_iterations = ipm_solve!(
        problem,
        inner_solver,
        options,
        time,
        x,
        previous_fval,
        iteration,
        inner_iterations
    )

    println("IPM solve completed with time: $time seconds.")
    println("Final solution: $x")
end

# Example 2: Solve a CUTEst problem using `ipm_solve!`
"""
This example demonstrates how to use `ipm_solve!` for a `CUTEstModel` problem.

# Expected Output:
    IPM solve completed with time: X seconds. # X≈6.12*10^-3 on my machine
    Final solution: [1.9999802700523392, 2.4171648142034597e-5]
"""
function example_ipm_solve_cutest()
    # Initialize a CUTEst problem
    problem = CUTEstModel("HS21")  # Example with constraints
    x = problem.meta.x0

    # Convert CUTEst problem to SEQUOIA
    sequoia_problem = cutest_to_sequoia(problem)
    sequoia_problem.solver_settings = SEQUOIA_Settings(
        :IntPt, 
        :LBFGS, 
        false, 
        1e-8, 
        100, 
        10.0, 
        1e-6; 
        solver_params = [1.0, 0.5, 0.1, 0, 0.0, 0.0, 0.0, 0.0, 0.0], 
        cost_tolerance = 1e-6, 
        store_trace = true
    )

    # Solver settings and options
    inner_solver = Optim.LBFGS()
    options = Optim.Options(iterations = 100, g_tol = 1e-6, store_trace=true, extended_trace=true)

    # Initialize variables
    time = 0.0
    previous_fval = obj(problem, x)
    iteration = 0
    inner_iterations = 0

    # Solve the IPM problem
    time, x, previous_fval, iteration, inner_iterations = ipm_solve!(
        sequoia_problem,
        inner_solver,
        options,
        time,
        x,
        previous_fval,
        iteration,
        inner_iterations
    )

    println("IPM solve completed with time: $time seconds.")
    println("Final solution: $x")

    finalize(problem)  # Finalize CUTEst environment
end

# Run Examples
example_ipm_solve_sequoia()
example_ipm_solve_cutest()
