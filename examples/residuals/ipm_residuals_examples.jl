using CUTEst
using NLPModels
using Sequoia
using LinearAlgebra

# Example 1: Compute the IPM objective for a SEQUOIA problem
"""
This example demonstrates how to compute the IPM objective for a `SEQUOIA_pb` problem.

# Expected Output:
    IPM objective value:
    66.26 
"""
function example_ipm_obj_sequoia()
    # Define a SEQUOIA problem
    problem = SEQUOIA_pb(
        3;
        x0 = [1.0, 2.0, 3.0],
        constraints = x -> [x[1] + x[2] - 5, x[3] - 0.5],
        jacobian = x -> [1.0 1.0 0.0; 0.0 0.0 1.0],
        eqcon = [1],
        ineqcon = [2],
        objective = x -> sum(x.^2),
        gradient = x -> 2 .* x
    )

    μ = 0.1  # Barrier parameter
    x_a = [problem.x0; zeros(length(problem.ineqcon) + length(problem.eqcon))]

    # Compute the IPM objective
    obj_value = ipm_obj(x_a, μ, problem)

    println("IPM objective value:")
    println(obj_value)
end

# Example 2: Compute the IPM gradient for a SEQUOIA problem
"""
This example demonstrates how to compute the IPM gradient for a `SEQUOIA_pb` problem.

# Expected Output:
    IPM gradient:
    [4.0, 12.0, 29.0, 12.0, 11.5]
"""
function example_ipm_grad_sequoia()
    # Define a SEQUOIA problem
    problem = SEQUOIA_pb(
        3;
        x0 = [1.0, 2.0, 3.0],
        constraints = x -> [x[1] + x[2] - 5, x[3] - 0.5],
        jacobian = x -> [1.0 1.0 0.0; 0.0 0.0 1.0],
        eqcon = [1],
        ineqcon = [2],
        objective = x -> sum(x.^2),
        gradient = x -> 2 .* x
    )

    μ = 0.1  # Barrier parameter
    x_a = [problem.x0; zeros(length(problem.ineqcon) + length(problem.eqcon))]
    grad_storage = zeros(length(x_a))

    # Compute the IPM gradient
    ipm_grad!(grad_storage, x_a, μ, problem)

    println("IPM gradient:")
    println(grad_storage)
end

# Example 3: Compute the IPM objective for a CUTEst problem
"""
This example demonstrates how to compute the IPM objective for a `CUTEstModel` problem.

# Expected Output:
    IPM objective value:
    374.0504
"""
function example_ipm_obj_cutest()
    # Initialize a CUTEst problem
    problem = CUTEstModel("HS21")  # Example problem with constraints
    x = problem.meta.x0

    μ = 0.1  # Barrier parameter
    nvar = problem.meta.nvar
    eq = length(problem.meta.jfix) + length(problem.meta.ifix)
    iq = length(problem.meta.jlow) + length(problem.meta.ilow) + length(problem.meta.jupp) + length(problem.meta.iupp) + 2 * (length(problem.meta.jrng) + length(problem.meta.irng))
    x_a = [x; zeros(eq + iq)]

    # Compute the IPM objective
    obj_value = ipm_obj(x_a, μ, problem)

    println("IPM objective value:")
    println(obj_value)

    finalize(problem)  # Finalize CUTEst environment
end

# Example 4: Compute the IPM gradient for a CUTEst problem
"""
This example demonstrates how to compute the IPM gradient for a `CUTEstModel` problem.

# Expected Output:
    IPM gradient:
    [-386.0008, 30.0, -7.4, -0.56, 13.8, 10.160000000000002, 6.200000000000001]
"""
function example_ipm_grad_cutest()
    # Initialize a CUTEst problem
    problem = CUTEstModel("HS21")  # Example problem with constraints
    x = problem.meta.x0

    μ = 0.1  # Barrier parameter
    nvar = problem.meta.nvar
    eq = length(problem.meta.jfix) + length(problem.meta.ifix)
    iq = length(problem.meta.jlow) + length(problem.meta.ilow) + length(problem.meta.jupp) + length(problem.meta.iupp) + 2 * (length(problem.meta.jrng) + length(problem.meta.irng))
    x_a = [x; zeros(eq + iq)]
    grad_storage = zeros(length(x_a))

    # Compute the IPM gradient
    ipm_grad!(grad_storage, x_a, μ, problem)

    println("IPM gradient:")
    println(grad_storage)

    finalize(problem)  # Finalize CUTEst environment
end

# Run Examples
example_ipm_obj_sequoia()
example_ipm_grad_sequoia()
example_ipm_obj_cutest()
example_ipm_grad_cutest()
