using Sequoia
using CUTEst
using Optim

# Example 1: Compute augmented Lagrangian objective for a SEQUOIA problem
"""
This example demonstrates how to compute the augmented Lagrangian objective for a `SEQUOIA_pb` problem.

# Expected Output:
    Augmented Lagrangian objective value:
    6.0
"""
function example_auglag_obj_sequoia()
    # Define a SEQUOIA problem
    problem = SEQUOIA_pb(
        2;
        x0 = [1.0, 2.0],
        constraints = x -> [x[1] + x[2] - 3, x[2] - 1],
        eqcon = [1],
        ineqcon = [2],
        jacobian = x -> [1.0 1.0; 0.0 1.0],
        objective = x -> sum(x.^2),
        gradient = x -> 2 .* x
    )
    
    μ = 1.0  # Penalty parameter
    λ = [0.5, 0.5]  # Lagrange multipliers
    x = problem.x0

    # Compute the augmented Lagrangian objective
    obj_value = auglag_obj(x, μ, λ, problem)

    println("Augmented Lagrangian objective value:")
    println(obj_value)
end

# Example 2: Compute the gradient of the augmented Lagrangian for a SEQUOIA problem
"""
This example demonstrates how to compute the gradient of the augmented Lagrangian objective for a `SEQUOIA_pb` problem.

# Expected Output:
    Gradient of the augmented Lagrangian:
    [2.0, 5.5]
"""
function example_auglag_grad_sequoia()
    # Define a SEQUOIA problem
    problem = SEQUOIA_pb(
        2;
        x0 = [1.0, 2.0],
        constraints = x -> [x[1] + x[2] - 3, x[2] - 1],
        eqcon = [1],
        ineqcon = [2],
        jacobian = x -> [1.0 1.0; 0.0 1.0],
        objective = x -> sum(x.^2),
        gradient = x -> 2 .* x
    )
    
    μ = 1.0  # Penalty parameter
    λ = [0.5, 0.5]  # Lagrange multipliers
    x = problem.x0
    grad_storage = zeros(length(x))  # Preallocate gradient storage

    # Compute the gradient of the augmented Lagrangian
    auglag_grad!(grad_storage, x, μ, λ, problem)

    println("Gradient of the augmented Lagrangian:")
    println(grad_storage)
end

# Example 3: Update Lagrange multipliers for a SEQUOIA problem
"""
This example demonstrates how to update the Lagrange multipliers for a `SEQUOIA_pb` problem.

# Expected Output:
    Updated Lagrange multipliers:
    [0.5, 1.5]
"""
function example_update_lag_mult_sequoia()
    # Define a SEQUOIA problem
    problem = SEQUOIA_pb(
        2;
        x0 = [1.0, 2.0],
        constraints = x -> [x[1] + x[2] - 3, x[2] - 1],
        eqcon = [1],
        ineqcon = [2],
        jacobian = x -> [1.0 1.0; 0.0 1.0],
        objective = x -> sum(x.^2),
        gradient = x -> 2 .* x
    )
    
    μ = 1.0  # Penalty parameter
    λ = [0.5, 0.5]  # Lagrange multipliers
    x = problem.x0

    # Update the Lagrange multipliers
    update_lag_mult!(x, μ, λ, problem)

    println("Updated Lagrange multipliers:")
    println(λ)
end

# Example 4: Compute augmented Lagrangian objective for a CUTEst problem
"""
This example demonstrates how to compute the augmented Lagrangian objective for a `CUTEstModel` problem.

# Expected Output:
    Augmented Lagrangian objective value:
    97.01
"""
function example_auglag_obj_cutest()
    # Initialize a CUTEst problem
    problem = CUTEstModel("HS21")  # Example problem with constraints
    x = problem.meta.x0
    μ = 1.0  # Penalty parameter
    λ = [0.5, 0.5, 0.5, 0.5, 0.5]  # Lagrange multipliers

    # Compute the augmented Lagrangian objective
    obj_value = auglag_obj(x, μ, λ, problem)

    println("Augmented Lagrangian objective value:")
    println(obj_value)

    finalize(problem)  # Finalize CUTEst environment
end

# Example 5: Compute the gradient of the augmented Lagrangian for a CUTEst problem
"""
This example demonstrates how to compute the gradient of the augmented Lagrangian objective for a `CUTEstModel` problem.

# Expected Output:
    Gradient of the augmented Lagrangian:
    [-1941.02, 189.0]
"""
function example_auglag_grad_cutest()
    # Initialize a CUTEst problem
    problem = CUTEstModel("HS21")  # Example problem with constraints
    x = problem.meta.x0  # Use the initial guess for x

    # Initialize the Lagrange multipliers
    total_eq_con = length(problem.meta.jfix) + length(problem.meta.ifix)
    total_ineq_con = length(problem.meta.jlow) + length(problem.meta.ilow) + 
                     length(problem.meta.jupp) + length(problem.meta.iupp) + 
                     2 * (length(problem.meta.jrng) + length(problem.meta.irng))
    λ = ones(total_eq_con + total_ineq_con)  # Initialize with ones
    μ = 10.0  # Penalty parameter

    # Preallocate the gradient vector
    grad_storage = zeros(length(x))

    # Compute the gradient of the augmented Lagrangian
    auglag_grad!(grad_storage, x, μ, λ, problem)

    println("Gradient of the augmented Lagrangian:")
    println(grad_storage)

    # Finalize the CUTEst environment
    finalize(problem)
end

# Example 6: Update Lagrange multipliers for a CUTEst problem
"""
This example demonstrates how to update the Lagrange multipliers for a `CUTEstModel` problem.

# Expected Output:
    Updated Lagrange multipliers:
    [19.0, 3.0, 0.0, 0.0, 0.0]
"""
function example_update_lag_mult_cutest()
    # Initialize a CUTEst problem
    problem = CUTEstModel("HS21")  # Example problem with constraints
    x = problem.meta.x0
    μ = 1.0  # Penalty parameter
    λ = zeros(length(res(x, problem)))  # Initialize multipliers

    # Update the Lagrange multipliers
    update_lag_mult!(x, μ, λ, problem)

    println("Updated Lagrange multipliers:")
    println(λ)

    finalize(problem)  # Finalize CUTEst environment
end

# Run Examples
example_auglag_obj_sequoia()
example_auglag_grad_sequoia()
example_update_lag_mult_sequoia()
example_auglag_obj_cutest()
example_auglag_grad_cutest()
example_update_lag_mult_cutest()
