using CUTEst
using NLPModels
using Sequoia
using LinearAlgebra

# Example 1: Compute the IPM objective for a SEQUOIA problem
"""
This example demonstrates how to compute the IPM objective for a `SEQUOIA_pb` problem.

# Expected Output:
    IPM objective value:
    66.3101 
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
    x_a = [problem.x0; zeros(length(problem.ineqcon) + length(problem.eqcon)); 0.1 * ones(length(problem.ineqcon))]

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
    [4.0, 12.0, 29.02, 12.0, 11.996, 1.004]
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
    x_a = [problem.x0; zeros(length(problem.ineqcon) + length(problem.eqcon)); 0.1 * ones(length(problem.ineqcon))]
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
    7974.4709
"""
function example_ipm_obj_cutest()
    # Initialize a CUTEst problem
    problem = CUTEstModel("HS21")  # Example problem with constraints
    x = problem.meta.x0

    μ = 0.1  # Barrier parameter
    nvar = problem.meta.nvar
    eq = length(problem.meta.jfix) + length(problem.meta.ifix)
    iq = length(problem.meta.jlow) + length(problem.meta.ilow) + length(problem.meta.jupp) + length(problem.meta.iupp) + 2 * (length(problem.meta.jrng) + length(problem.meta.irng))
    x_a = [x; zeros(eq + iq); 0.1 * ones(iq)]

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
    [-488.2008000000001, 26.019999999999996, -3.604, 0.036, 3.996, -0.044000000000000004, -4.004, 7.604000000000001, 1.204, -19.596000000000004, -20.396, -20.396]
"""
function example_ipm_grad_cutest()
    # Initialize a CUTEst problem
    problem = CUTEstModel("HS21")  # Example problem with constraints
    x = problem.meta.x0

    μ = 0.1  # Barrier parameter
    nvar = problem.meta.nvar
    eq = length(problem.meta.jfix) + length(problem.meta.ifix)
    iq = length(problem.meta.jlow) + length(problem.meta.ilow) + length(problem.meta.jupp) + length(problem.meta.iupp) + 2 * (length(problem.meta.jrng) + length(problem.meta.irng))
    x_a = [x; zeros(eq + iq); 0.1 * ones(iq)]
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
