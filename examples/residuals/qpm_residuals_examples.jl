# Example Use Cases for `qpm_obj` and `qpm_grad!`
using CUTEst

# Example 1: Compute the quadratic penalty objective for a SEQUOIA problem
"""
This example demonstrates how to compute the quadratic penalty objective function for a `SEQUOIA_pb` problem.

# Usage:
    example_qpm_obj_sequoia()

# Expected Output:
    Quadratic penalty objective value:
    12.5
"""
function example_qpm_obj_sequoia()
    # Create a SEQUOIA problem instance
    problem = SEQUOIA_pb(
        3;
        x0 = [1.0, 2.0, 3.0],
        objective = x -> sum(x.^2),
        gradient = x -> 2 .* x,
        constraints = x -> [x[1] + x[2] - 5, x[2] - x[3] - 1],
        eqcon = [1],
        ineqcon = [2],
        jacobian = x -> [1.0 1.0 0.0; 0.0 1.0 -1.0]
    )
    x = problem.x0
    μ = 2.5

    # Compute the quadratic penalty objective
    qpm_value = qpm_obj(x, μ, problem)

    println("Quadratic penalty objective value:")
    println(qpm_value)
end

# Example 2: Compute the gradient of the quadratic penalty objective for a SEQUOIA problem
"""
This example demonstrates how to compute the gradient of the quadratic penalty objective function for a `SEQUOIA_pb` problem.

# Usage:
    example_qpm_grad_sequoia()

# Expected Output:
    Gradient of the quadratic penalty objective:
    [8.0, 14.0, 6.0]
"""
function example_qpm_grad_sequoia()
    # Create a SEQUOIA problem instance
    problem = SEQUOIA_pb(
        3;
        x0 = [1.0, 2.0, 3.0],
        objective = x -> sum(x.^2),
        gradient = x -> 2 .* x,
        constraints = x -> [x[1] + x[2] - 5, x[2] - x[3] - 1],
        eqcon = [1],
        ineqcon = [2],
        jacobian = x -> [1.0 1.0 0.0; 0.0 1.0 -1.0]
    )
    x = problem.x0
    μ = 2.5
    g = zeros(length(x))

    # Compute the gradient of the quadratic penalty objective
    qpm_grad!(g, x, μ, problem)

    println("Gradient of the quadratic penalty objective:")
    println(g)
end

# Example 3: Compute the quadratic penalty objective for a CUTEst problem
"""
This example demonstrates how to compute the quadratic penalty objective function for a `CUTEstModel` problem.

# Usage:
    example_qpm_obj_cutest()

# Expected Output:
    Quadratic penalty objective value:
    98.25
"""
function example_qpm_obj_cutest()
    # Initialize a CUTEst problem
    problem = CUTEstModel("HS21")  # Example with constraints
    x = problem.meta.x0
    μ = 2.0

    # Compute the quadratic penalty objective
    qpm_value = qpm_obj(x, μ, problem)

    println("Quadratic penalty objective value:")
    println(qpm_value)

    finalize(problem)
end

# Example 4: Compute the gradient of the quadratic penalty objective for a CUTEst problem
"""
This example demonstrates how to compute the gradient of the quadratic penalty objective function for a `CUTEstModel` problem.

# Usage:
    example_qpm_grad_cutest()

# Expected Output:
    Gradient of the quadratic penalty objective:
    [-25.0, 50.0]
"""
function example_qpm_grad_cutest()
    # Initialize a CUTEst problem
    problem = CUTEstModel("HS21")  # Example with constraints
    x = problem.meta.x0
    μ = 2.0
    g = zeros(length(x))

    # Compute the gradient of the quadratic penalty objective
    qpm_grad!(g, x, μ, problem)

    println("Gradient of the quadratic penalty objective:")
    println(g)

    finalize(problem)
end

# Call examples
example_qpm_obj_sequoia()
example_qpm_grad_sequoia()
example_qpm_obj_cutest()
example_qpm_grad_cutest()
