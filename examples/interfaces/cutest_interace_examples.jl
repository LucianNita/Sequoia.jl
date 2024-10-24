"""
# Example: Converting a Constrained CUTEst Problem to `SEQUOIA_pb`

This example demonstrates how to load a constrained problem from the CUTEst library (specifically "HS21"), 
convert it into a `SEQUOIA_pb` instance, and access the relevant information such as objective and constraint 
details.

```julia
example_cutest_to_sequoia()

# Load a constrained CUTEst problem
cutest_problem = CUTEstModel("HS21")

# Convert the problem to a SEQUOIA_pb instance
sequoia_problem = cutest_to_sequoia(cutest_problem)

# Print general information about the problem
println("Number of Variables: ", sequoia_problem.nvar)
println("Initial Guess: ", sequoia_problem.x0)
println("Objective Value at x0: ", sequoia_problem.objective(sequoia_problem.x0))

# Access constraint-related data
if sequoia_problem.constraints !== nothing
    println("Constraints at x0: ", sequoia_problem.constraints(sequoia_problem.x0))
    if sequoia_problem.jacobian !== nothing
        println("Jacobian at x0: ", sequoia_problem.jacobian(sequoia_problem.x0))
    end
end

#Finalize the problem
finalize(cutest_problem)

Example output: 

Number of Variables: 2
Initial Guess: [-1.0, -1.0]
Objective Value at x0: -98.99
Constraints at x0: [-19.0]
Jacobian at x0: sparse([1, 1], [1, 2], [10.0, -1.0], 1, 2)

"""
function example_cutest_to_sequoia()
    # Load a constrained CUTEst problem
    cutest_problem = CUTEstModel("HS21")

    # Convert the problem to a SEQUOIA_pb instance
    sequoia_problem = cutest_to_sequoia(cutest_problem)

    # Print general information about the problem
    println("Number of Variables: ", sequoia_problem.nvar)
    println("Initial Guess: ", sequoia_problem.x0)
    println("Objective Value at x0: ", sequoia_problem.objective(sequoia_problem.x0))

    # Access constraint-related data
    if sequoia_problem.constraints !== nothing
        println("Constraints at x0: ", sequoia_problem.constraints(sequoia_problem.x0))
        if sequoia_problem.jacobian !== nothing
            println("Jacobian at x0: ", sequoia_problem.jacobian(sequoia_problem.x0))
        end
    end

    #Finalize the problem
    finalize(cutest_problem)
end
