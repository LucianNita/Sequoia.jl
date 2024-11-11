"""
# cutest_to_sequoia Examples
# This file contains example use cases for the `cutest_to_sequoia` function, demonstrating its usage in different scenarios:
# 1. Conversion of an unconstrained CUTEst problem to a `SEQUOIA_pb` instance.
# 2. Conversion of a CUTEst problem with equality constraints.
# 3. Conversion of a CUTEst problem with inequality constraints.
# 4. Conversion of a CUTEst problem with both equality and inequality constraints.
# 5. Handling large-scale CUTEst problems.
# 6. Validating the output `SEQUOIA_pb` instance.
"""

using Sequoia
using CUTEst

# Example 1: Unconstrained CUTEst Problem
"""
This example demonstrates how to convert an unconstrained CUTEst problem into a `SEQUOIA_pb` instance.

# Usage:
    example_unconstrained_cutest()

# Expected Output:
    SEQUOIA_pb instance with no constraints and default objective/gradient settings.
"""
function example_unconstrained_cutest()
    cutest_problem = CUTEstModel("ROSENBR")  # Rosenbrock function
    pb = cutest_to_sequoia(cutest_problem)
    println("Converted SEQUOIA_pb instance:")
    println(pb)
    finalize(cutest_problem)
end

# Example 2: Equality-Constrained CUTEst Problem
"""
This example demonstrates how to convert a CUTEst problem with equality constraints into a `SEQUOIA_pb` instance.

# Usage:
    example_equality_constrained_cutest()

# Expected Output:
    SEQUOIA_pb instance with equality constraints and their Jacobian.
"""
function example_equality_constrained_cutest()
    cutest_problem = CUTEstModel("HS28")  # Example with equality constraints
    pb = cutest_to_sequoia(cutest_problem)
    println("Converted SEQUOIA_pb instance with equality constraints:")
    println(pb)
    println("Equality constraints indices: ", pb.eqcon)
    finalize(cutest_problem)
end

# Example 3: Inequality-Constrained CUTEst Problem
"""
This example demonstrates how to convert a CUTEst problem with inequality constraints into a `SEQUOIA_pb` instance.

# Usage:
    example_inequality_constrained_cutest()

# Expected Output:
    SEQUOIA_pb instance with inequality constraints and their Jacobian.
"""
function example_inequality_constrained_cutest()
    cutest_problem = CUTEstModel("HS76")  # Example with inequality constraints
    pb = cutest_to_sequoia(cutest_problem)
    println("Converted SEQUOIA_pb instance with inequality constraints:")
    println(pb)
    println("Inequality constraints indices: ", pb.ineqcon)
    finalize(cutest_problem)
end

# Example 4: Mixed Constraints in CUTEst Problem
"""
This example demonstrates how to convert a CUTEst problem with both equality and inequality constraints into a `SEQUOIA_pb` instance.

# Usage:
    example_mixed_constraints_cutest()

# Expected Output:
    SEQUOIA_pb instance with both equality and inequality constraints, and their Jacobians.
"""
function example_mixed_constraints_cutest()
    cutest_problem = CUTEstModel("HS75")  # Example with mixed constraints
    pb = cutest_to_sequoia(cutest_problem)
    println("Converted SEQUOIA_pb instance with mixed constraints:")
    println(pb)
    println("Equality constraints indices: ", pb.eqcon)
    println("Inequality constraints indices: ", pb.ineqcon)
    finalize(cutest_problem)
end

# Example 5: Large-Scale CUTEst Problem
"""
This example demonstrates how to convert a large-scale CUTEst problem into a `SEQUOIA_pb` instance.

# Usage:
    example_large_scale_cutest()

# Expected Output:
    SEQUOIA_pb instance with performance-validated objective, gradient, and constraints.
"""
function example_large_scale_cutest()
    cutest_problem = CUTEstModel("STNQP1")  # Large-scale problem
    pb = cutest_to_sequoia(cutest_problem)
    println("Converted SEQUOIA_pb instance for large-scale problem:")
    println(pb)
    finalize(cutest_problem)
end

# Example 6: Validate Converted `SEQUOIA_pb` Instance
"""
This example demonstrates how to validate the `SEQUOIA_pb` instance created from a CUTEst problem.

# Usage:
    example_validate_sequoia_instance()

# Expected Output:
    Validation successful for the converted SEQUOIA_pb instance.
"""
function example_validate_sequoia_instance()
    cutest_problem = CUTEstModel("HS35")  # Example problem
    pb = cutest_to_sequoia(cutest_problem)
    println("Validating converted SEQUOIA_pb instance...")
    validate_pb!(pb)
    println("Validation successful!")
    finalize(cutest_problem)
end

# Run all examples
function run_all_examples()
    example_unconstrained_cutest()
    example_equality_constrained_cutest()
    example_inequality_constrained_cutest()
    example_mixed_constraints_cutest()
    example_large_scale_cutest()
    example_validate_sequoia_instance()
end

run_all_examples()
