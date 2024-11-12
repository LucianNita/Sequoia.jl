using Optim
using Sequoia
using CUTEst
using NLPModels

# Example 1: Feasibility solve for a SEQUOIA problem
"""
This example demonstrates how to use `feasibility_solve!` for a `SEQUOIA_pb` problem.

# Expected Output:
    Feasibility solve completed with time: X seconds. #On my machine X≈6.29*10^-5, but depends on run, computer etc. (note first run is slower, second run matters)
    Final solution: [2.0, 3.0, 0.4061990212071787]
"""
function example_feasibility_solve_sequoia()
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
    x = problem.x0
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
    println("Feasibility solve completed with time: $time seconds.")
    println("Final solution: $x")
end

# Example 2: Feasibility solve for a CUTEst problem
"""
This example demonstrates how to use `feasibility_solve!` for a `CUTEstModel` problem.

# Expected Output:
    Feasibility solve completed with time: X seconds. #On my machine X≈3.69*10^-5, but depends on run, computer etc. (note here cutest takes care of precompilation & averaging so first run will be fast still) 
    Final solution: [50.0, -6.020725388601036]
"""
function example_feasibility_solve_cutest()
    # Initialize a CUTEst problem
    problem = CUTEstModel("HS21")
    x = problem.meta.x0
    inner_solver = Optim.LBFGS()
    options = Optim.Options(iterations = 100, g_tol = 1e-6)

    # Wrap CUTEst problem in SEQUOIA_pb
    sequioa_problem = cutest_to_sequoia(problem)

    # Initialize variables
    time = 0.0
    previous_fval = obj(problem, x)
    inner_iterations = 0

    # Solve the feasibility problem
    time, x, previous_fval, inner_iterations = feasibility_solve!(
        sequioa_problem,
        inner_solver,
        options,
        time,
        x,
        previous_fval,
        inner_iterations
    )

    println("Feasibility solve completed with time: $time seconds.")
    println("Final solution: $x")

    finalize(problem)  # Finalize CUTEst environment
end

# Run the examples
example_feasibility_solve_sequoia()
example_feasibility_solve_cutest()
