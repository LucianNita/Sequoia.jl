using CUTEst
using Sequoia

# Example 1: Compute penalty function for a CUTEst problem
"""
This example demonstrates how to compute the penalty function `r(x, t_k)` for a `CUTEstModel` problem.

# Usage:
    example_r_cutest()

# Expected Output:
    Penalty function value:
    185.0
"""
function example_r_cutest()
    # Initialize a CUTEst problem
    problem = CUTEstModel("HS21")  # Example with constraints
    x = problem.meta.x0  # Use the initial guess for x
    tk = 10.0  # Threshold value for the objective

    # Compute the penalty function
    penalty_value = r(x, tk, problem)

    println("Penalty function value:")
    println(penalty_value)

    finalize(problem)  # Finalize CUTEst environment
end

# Example 2: Compute gradient of penalty function for a CUTEst problem
"""
This example demonstrates how to compute the gradient of the penalty function `r(x, t_k)` for a `CUTEstModel` problem.

# Usage:
    example_r_gradient_cutest()

# Expected Output:
    Gradient of the penalty function:
    [-193.0, 19.0]
"""
function example_r_gradient_cutest()
    # Initialize a CUTEst problem
    problem = CUTEstModel("HS21")  # Example with constraints
    x = problem.meta.x0  # Use the initial guess for x
    tk = 10.0  # Threshold value for the objective
    grad_storage = zeros(length(x))  # Preallocate gradient storage

    # Compute the gradient of the penalty function
    r_gradient!(grad_storage, x, tk, problem)

    println("Gradient of the penalty function:")
    println(grad_storage)

    finalize(problem)  # Finalize CUTEst environment
end

# Example 3: Compute penalty function for a SEQUOIA problem
"""
This example demonstrates how to compute the penalty function `r(x, t_k)` for a `SEQUOIA_pb` problem.

# Usage:
    example_r_sequoia()

# Expected Output:
    Penalty function value:
    10.0
"""
function example_r_sequoia()
    # Create a SEQUOIA problem instance
    problem = SEQUOIA_pb(
        3;
        x0 = [1.0, 2.0, 3.0],
        objective = x -> sum(x.^2),
        gradient = x -> 2 .* x,
        constraints = x -> [x[1] + x[2] - 5],
        jacobian = x -> [1.0 1.0 0.0],
        eqcon = [1],
        ineqcon = Int[]
    )
    x = problem.x0  # Use the initial guess for x
    tk = 10.0  # Threshold value for the objective

    # Compute the penalty function
    penalty_value = r(x, tk, problem)

    println("Penalty function value:")
    println(penalty_value)
end

# Example 4: Compute gradient of penalty function for a SEQUOIA problem
"""
This example demonstrates how to compute the gradient of the penalty function `r(x, t_k)` for a `SEQUOIA_pb` problem.

# Usage:
    example_r_gradient_sequoia()

# Expected Output:
    Gradient of the penalty function:
    [6.0, 14.0, 24.0]
"""
function example_r_gradient_sequoia()
    # Create a SEQUOIA problem instance
    problem = SEQUOIA_pb(
        3;
        x0 = [1.0, 2.0, 3.0],
        objective = x -> sum(x.^2),
        gradient = x -> 2 .* x,
        constraints = x -> [x[1] + x[2] - 5],
        jacobian = x -> [1.0 1.0 0.0],
        eqcon = [1],
        ineqcon = Int[]
    )
    x = problem.x0  # Use the initial guess for x
    tk = 10.0  # Threshold value for the objective
    grad_storage = zeros(length(x))  # Preallocate gradient storage

    # Compute the gradient of the penalty function
    r_gradient!(grad_storage, x, tk, problem)

    println("Gradient of the penalty function:")
    println(grad_storage)
end

# Run examples
example_r_cutest()
example_r_gradient_cutest()
example_r_sequoia()
example_r_gradient_sequoia()
