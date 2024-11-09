"""
# SEQUOIA_Solution_step Examples

This file contains example use cases for the `SEQUOIA_Solution_step` struct, demonstrating its usage in different scenarios:
1. A successful iteration.
2. Iteration terminated due to small step size.
3. Handling an unbounded problem.
4. Handling an infeasible problem.
5. Debugging a solver step.
6. Minimal example with default fields.
"""

# Example 1: A Successful Iteration
"""
This example represents a successful optimization iteration where the solver meets the first-order optimality conditions.

# Usage:
    example_successful_iteration()

# Expected Output:
    SEQUOIA_Solution_step(5, 1.0e-8, :first_order, 0.02, 2, [1.0, 2.0], -5.0, [0.0, 0.0], nothing, [0.5, 0.3], [[0.8, 1.5], [0.9, 1.8], [1.0, 2.0]])
"""
function example_successful_iteration()
    solution = SEQUOIA_Solution_step(
        5,                      # Outer iteration number
        1e-8,                   # Convergence metric
        :first_order,           # Solver status
        0.02,                   # Inner computation time
        2,                      # Number of inner iterations
        [1.0, 2.0],             # Solution vector
        -5.0,                   # Objective value
        [0.0, 0.0],             # Gradient vector
        nothing,                # No constraints
        [0.5, 0.3],             # Solver parameters
        [[0.8, 1.5], [0.9, 1.8], [1.0, 2.0]] # Inner x iterates history
    )
    println(solution)
end

# Example 2: Iteration Terminated Due to Small Step Size
"""
This example illustrates an iteration where the solver terminates because the step size becomes too small.

# Usage:
    example_small_step_termination()

# Expected Output:
    SEQUOIA_Solution_step(8, 0.0001, :small_step, 0.05, 12, [0.99, 1.01], -4.8, [0.01, -0.01], [0.02], [0.8, 0.5], nothing)
"""
function example_small_step_termination()
    solution = SEQUOIA_Solution_step(
        8,                      # Outer iteration number
        1e-4,                   # Convergence metric
        :small_step,            # Solver status
        0.05,                   # Inner computation time
        12,                     # Number of inner iterations
        [0.99, 1.01],           # Solution vector
        -4.8,                   # Objective value
        [0.01, -0.01],          # Gradient vector
        [0.02],                 # Constraints
        [0.8, 0.5],             # Solver parameters
        nothing                 # No x iterates history
    )
    println(solution)
end

# Example 3: Handling an Unbounded Problem
"""
This example shows a scenario where the solver determines that the optimization problem is unbounded.

# Usage:
    example_unbounded_problem()

# Expected Output:
    SEQUOIA_Solution_step(15, -Inf, :unbounded, 0.1, 20, [100.0, -100.0], Inf, [10.0, -10.0], nothing, [1.0, 1.5], nothing)
"""
function example_unbounded_problem()
    solution = SEQUOIA_Solution_step(
        15,                     # Outer iteration number
        Inf,                    # Convergence metric indicating divergence
        :unbounded,             # Solver status
        0.1,                    # Inner computation time
        20,                     # Number of inner iterations
        [100.0, -100.0],        # Solution vector trending towards infinity
        -Inf,                   # Objective value diverging
        [10.0, -10.0],          # Gradient vector
        nothing,                # No constraints
        [1.0, 1.5],             # Solver parameters
        nothing                 # No x iterates history
    )
    println(solution)
end

# Example 4: Handling an Infeasible Problem
"""
This example represents an infeasible problem where the solver cannot find a solution that satisfies the constraints.

# Usage:
    example_infeasible_problem()

# Expected Output:
    SEQUOIA_Solution_step(20, 1.0, :infeasible, 0.2, 15, [0.5, 0.5], 10.0, [2.0, 2.0], [1.0, -1.5], nothing, nothing)
"""
function example_infeasible_problem()
    solution = SEQUOIA_Solution_step(
        20,                     # Outer iteration number
        1.0,                    # Convergence metric
        :infeasible,            # Solver status
        0.2,                    # Inner computation time
        15,                     # Number of inner iterations
        [0.5, 0.5],             # Solution vector
        10.0,                   # Objective value
        [2.0, 2.0],             # Gradient vector
        [1.0, -1.5],            # Constraints
        nothing,                # No solver parameters
        nothing                 # No x iterates history
    )
    println(solution)
end

# Example 5: Debugging a Solver Step
"""
This example demonstrates how to use `SEQUOIA_Solution_step` to debug solver behavior by examining all fields in detail.

# Usage:
    example_debugging_solver_step()

# Expected Output:
    SEQUOIA_Solution_step(3, 0.001, :acceptable, 0.15, 8, [1.2, 1.8], -3.5, [0.02, -0.03], [0.0, 0.01], [0.7, 0.4], [[1.1, 1.7], [1.15, 1.75], [1.2, 1.8]])
"""
function example_debugging_solver_step()
    solution = SEQUOIA_Solution_step(
        3,                      # Outer iteration number
        1e-3,                   # Convergence metric
        :acceptable,            # Solver status
        0.15,                   # Inner computation time
        2,                      # Number of inner iterations
        [1.2, 1.8],             # Solution vector
        -3.5,                   # Objective value
        [0.02, -0.03],          # Gradient vector
        [0.0, 0.01],            # Constraints
        [0.7, 0.4],             # Solver parameters
        [[1.1, 1.7], [1.15, 1.75], [1.2, 1.8]] # x iterates history
    )
    println(solution)
end

# Example 6: Minimal Example with Default Fields
"""
This minimal example shows how to define a `SEQUOIA_Solution_step` instance with only the required fields.

# Usage:
    example_minimal_solution_step()

# Expected Output:
    SEQUOIA_Solution_step(1, 0.01, :max_iter, 0.05, 5, [0.0, 1.0], 0.2, [0.1, -0.1], nothing, nothing, nothing)
"""
function example_minimal_solution_step()
    solution = SEQUOIA_Solution_step(
        1,                      # Outer iteration number
        0.01,                   # Convergence metric
        :max_iter,              # Solver status
        0.05,                   # Inner computation time
        5,                      # Number of inner iterations
        [0.0, 1.0],             # Solution vector
        0.2,                    # Objective value
        [0.1, -0.1]             # Gradient vector
    )
    println(solution)
end

using Sequoia

# Call All Examples
example_successful_iteration()
example_small_step_termination()
example_unbounded_problem()
example_infeasible_problem()
example_debugging_solver_step()
example_minimal_solution_step()
