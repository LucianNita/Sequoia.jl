# Example Use Cases for `r0` and `r0_gradient!`
using CUTEst
using Sequoia

# Example 1: Compute residual for a CUTEst problem
"""
This example demonstrates how to compute the residual function `r0` for a `CUTEstModel` problem.

# Usage:
    example_r0_cutest()

# Expected Output:
    Residual value:
    185.0   
"""
function example_r0_cutest()
    # Initialize a CUTEst problem
    problem = CUTEstModel("HS21")  # Example with constraints
    x = problem.meta.x0  # Use the initial guess for x

    # Compute the residual
    residual = r0(x, problem)

    println("Residual value:")
    println(residual)

    finalize(problem)  # Finalize CUTEst environment
end

# Example 2: Compute gradient of the residual for a CUTEst problem
"""
This example demonstrates how to compute the gradient of the residual function `r0` for a `CUTEstModel` problem.

# Usage:
    example_r0_gradient_cutest()

# Expected Output:
    Gradient of the residual:
    [-193.0, 19.0]
"""
function example_r0_gradient_cutest()
    # Initialize a CUTEst problem
    problem = CUTEstModel("HS21")  # Example with constraints
    x = problem.meta.x0  # Use the initial guess for x
    g = zeros(length(x))  # Preallocate gradient vector

    # Compute the gradient
    r0_gradient!(g, x, problem)

    println("Gradient of the residual:")
    println(g)

    finalize(problem)  # Finalize CUTEst environment
end

# Example 3: Compute residual for a SEQUOIA problem
"""
This example demonstrates how to compute the residual function `r0` for a `SEQUOIA_pb` problem.

# Usage:
    example_r0_sequoia()

# Expected Output:
    Residual value:
    4.0
"""
function example_r0_sequoia()
    # Create a SEQUOIA problem instance
    problem = SEQUOIA_pb(
        3;
        x0 = [1.0, 2.0, 3.0],
        constraints = x -> [x[1] + x[2] - 5, x[2] - x[3] - 1],
        eqcon = [1],
        ineqcon = [2],
        jacobian = x -> [1.0 1.0 0.0; 0.0 1.0 -1.0]
    )
    x = problem.x0  # Use the initial guess for x

    # Compute the residual
    residual = r0(x, problem)

    println("Residual value:")
    println(residual)
end

# Example 4: Compute gradient of the residual for a SEQUOIA problem
"""
This example demonstrates how to compute the gradient of the residual function `r0` for a `SEQUOIA_pb` problem.

# Usage:
    example_r0_gradient_sequoia()

# Expected Output:
    Gradient of the residual:
    [-2.0, -2.0, 0.0]
"""
function example_r0_gradient_sequoia()
    # Create a SEQUOIA problem instance
    problem = SEQUOIA_pb(
        3;
        x0 = [1.0, 2.0, 3.0],
        constraints = x -> [x[1] + x[2] - 5, x[2] - x[3] - 1],
        eqcon = [1],
        ineqcon = [2],
        jacobian = x -> [1.0 1.0 0.0; 0.0 1.0 -1.0]
    )
    x = problem.x0  # Use the initial guess for x
    g = zeros(length(x))  # Preallocate gradient vector

    # Compute the gradient
    r0_gradient!(g, x, problem)

    println("Gradient of the residual:")
    println(g)
end

# Run examples
example_r0_cutest()
example_r0_gradient_cutest()
example_r0_sequoia()
example_r0_gradient_sequoia()
